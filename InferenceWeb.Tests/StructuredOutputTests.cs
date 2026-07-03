using System.Text;
using System.Text.Json;

namespace InferenceWeb.Tests;

public class StructuredOutputTests
{
    // Stream a model output in many small fragments through the json_object
    // streaming repairer and return the concatenation actually sent to the
    // client, including the end-of-stream closing suffix.
    private static string FeedInChunks(string modelOutput, int chunkSize = 3)
    {
        var filter = new StreamingJsonObjectRepairer();
        var sb = new StringBuilder();
        for (int i = 0; i < modelOutput.Length; i += chunkSize)
            sb.Append(filter.Feed(modelOutput.Substring(i, System.Math.Min(chunkSize, modelOutput.Length - i))));
        sb.Append(filter.Finish());
        return sb.ToString();
    }

    [Fact]
    public void StreamingJsonFilterStripsCodeFencesAndTrailingTags()
    {
        // Exactly the messy shape observed live (markdown fence + a leaked
        // Gemma channel tag after the object).
        string raw = "```json\n{\n  \"name\": \"Mars\",\n  \"diameter_km\": 6779,\n  \"has_moons\": true\n}\n```<channel|>";
        string streamed = FeedInChunks(raw);

        using var doc = JsonDocument.Parse(streamed); // must be valid JSON
        Assert.Equal("Mars", doc.RootElement.GetProperty("name").GetString());
        Assert.DoesNotContain("```", streamed, System.StringComparison.Ordinal);
        Assert.DoesNotContain("channel", streamed, System.StringComparison.Ordinal);
    }

    [Fact]
    public void StreamingJsonFilterKeepsBracesInsideStrings()
    {
        string raw = "prefix {\"text\": \"a } b { c\", \"n\": 1} trailing";
        string streamed = FeedInChunks(raw, chunkSize: 1);

        using var doc = JsonDocument.Parse(streamed);
        Assert.Equal("a } b { c", doc.RootElement.GetProperty("text").GetString());
        Assert.Equal("""{"text": "a } b { c", "n": 1}""", streamed);
    }

    [Fact]
    public void StreamingJsonFilterStopsAtFirstBalancedObject()
    {
        var filter = new StreamingJsonObjectRepairer();
        string emitted = filter.Feed("{\"a\":1}{\"b\":2}");
        Assert.Equal("""{"a":1}""", emitted);
        Assert.True(filter.Done);
        Assert.Equal("", filter.Feed("more text")); // nothing after close
        Assert.Equal("", filter.Finish());          // nothing left to close
    }

    [Fact]
    public void StreamingJsonRepairerClosesTruncatedOutput()
    {
        // max_tokens hit in the middle of a string value
        string streamed = FeedInChunks("{\"name\": \"Ma");

        Assert.Equal("""{"name": "Ma"}""", streamed);
        using var doc = JsonDocument.Parse(streamed);
        Assert.Equal("Ma", doc.RootElement.GetProperty("name").GetString());
    }

    [Fact]
    public void StreamingJsonRepairerCompletesDanglingKeysValuesAndLiterals()
    {
        // truncated right after a key
        Assert.Equal("""{"a":null}""", FeedInChunks("{\"a\""));
        // truncated right after the colon
        Assert.Equal("""{"a": null}""", FeedInChunks("{\"a\": "));
        // truncated inside a literal and inside a nested array
        string streamed = FeedInChunks("{\"items\": [1, 2, {\"ok\": tru");
        Assert.Equal("""{"items": [1, 2, {"ok": true}]}""", streamed);
        JsonDocument.Parse(streamed).Dispose();
        // truncated number
        Assert.Equal("""{"n": 12.0}""", FeedInChunks("{\"n\": 12."));
    }

    [Fact]
    public void StreamingJsonRepairerFixesSingleQuotesUnquotedKeysAndPythonLiterals()
    {
        string streamed = FeedInChunks("{name: 'Mars', ok: True, note: None, bad: NaN}", chunkSize: 1);

        using var doc = JsonDocument.Parse(streamed);
        Assert.Equal("Mars", doc.RootElement.GetProperty("name").GetString());
        Assert.True(doc.RootElement.GetProperty("ok").GetBoolean());
        Assert.Equal(JsonValueKind.Null, doc.RootElement.GetProperty("note").ValueKind);
        Assert.Equal(JsonValueKind.Null, doc.RootElement.GetProperty("bad").ValueKind);
    }

    [Fact]
    public void StreamingJsonRepairerDropsTrailingCommasAndInsertsMissingOnes()
    {
        string trailing = FeedInChunks("{\"a\": [1, 2,], \"b\": 2,}");
        using (var doc = JsonDocument.Parse(trailing))
            Assert.Equal(2, doc.RootElement.GetProperty("a").GetArrayLength());

        string missing = FeedInChunks("{\"a\": 1 \"b\": 2}");
        using (var doc = JsonDocument.Parse(missing))
            Assert.Equal(2, doc.RootElement.GetProperty("b").GetInt32());
    }

    [Fact]
    public void StreamingJsonRepairerEscapesRawControlCharactersInStrings()
    {
        string streamed = FeedInChunks("{\"text\": \"line1\nline2\"}");

        using var doc = JsonDocument.Parse(streamed);
        Assert.Equal("line1\nline2", doc.RootElement.GetProperty("text").GetString());
    }

    [Fact]
    public void StreamingJsonRepairerFallsBackToWrappedProseOrEmptyObject()
    {
        string wrapped = FeedInChunks("I cannot answer that.");
        using (var doc = JsonDocument.Parse(wrapped))
            Assert.Equal("I cannot answer that.", doc.RootElement.GetProperty("response").GetString());

        var filter = new StreamingJsonObjectRepairer();
        Assert.Equal("{}", filter.Finish());
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(64)]
    public void StreamingJsonRepairerAlwaysYieldsParseableJsonRegardlessOfChunking(int chunkSize)
    {
        string[] nastyOutputs =
        {
            "```json\n{\"a\": [1, 2, {\"b\": 'x'}],}\n``` Done!",
            "{\"a\": \"unterminated",
            "{\"k\": +012.50, \"e\": 1e}",
            "{\"nested\": {\"deep\": [true, False, None, NaN",
            "{'single': 'quotes', \"mix\": \"double\"}",
            "{\"a\"",
            "{\"a\":}",
            "{\"a\": \"b} c\", \"d\": 2}",
            "<|channel|>final {\"x\": 1}<|end|>",
            "{\"a\" 1, \"b\":2}",
            "{\"a\": \"x\" \"y\": 2}",
            "{\"esc\": \"bad \\q escape\", \"u\": \"\\u12ZZ\"}",
            "{\"a\": [1, [2, {\"b\": [3",
            "{,}",
            "{\"a\": hello world}",
        };

        foreach (string raw in nastyOutputs)
        {
            string streamed = FeedInChunks(raw, chunkSize);
            using var doc = JsonDocument.Parse(streamed);
            Assert.Equal(JsonValueKind.Object, doc.RootElement.ValueKind);
        }
    }

    [Fact]
    public void StreamingJsonRepairerSurvivesRandomizedFuzzInput()
    {
        // Deterministic pseudo-fuzz: random JSON-ish garbage must still come out
        // as exactly one parseable JSON object, no matter how it is chunked.
        var rng = new System.Random(20260703);
        const string alphabet = "{}[]\":',.\\ \t\n0123456789eE+-truefalsnTFNIxé中";

        for (int iteration = 0; iteration < 2000; iteration++)
        {
            int length = rng.Next(0, 120);
            var raw = new StringBuilder(length);
            for (int i = 0; i < length; i++)
                raw.Append(alphabet[rng.Next(alphabet.Length)]);

            string input = raw.ToString();
            string streamed = FeedInChunks(input, chunkSize: rng.Next(1, 9));

            try
            {
                using var doc = JsonDocument.Parse(streamed);
                Assert.Equal(JsonValueKind.Object, doc.RootElement.ValueKind);
            }
            catch (JsonException ex)
            {
                Assert.Fail($"Iteration {iteration} produced unparseable JSON.\nInput: {input}\nOutput: {streamed}\nError: {ex.Message}");
            }
        }
    }


    [Fact]
    public void Qwen35NoThinkingTemplateKeepsPriorAnswerAsNextTurnPrefix()
    {
        const string jinjaTemplate = "{{ 'from-jinja' }}";

        var turn1 = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What is the tallest mountain in the world?" }
        };
        string renderedTurn1 = ChatTemplate.RenderFromGgufTemplate(
            jinjaTemplate, turn1, addGenerationPrompt: true, architecture: "qwen35", enableThinking: false);

        const string answer = "Mount Everest";
        var turn2 = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What is the tallest mountain in the world?" },
            new() { Role = "assistant", Content = answer },
            new() { Role = "user", Content = "How tall is it in meters?" }
        };
        string renderedTurn2 = ChatTemplate.RenderFromGgufTemplate(
            jinjaTemplate, turn2, addGenerationPrompt: true, architecture: "qwen35", enableThinking: false);

        Assert.DoesNotContain("from-jinja", renderedTurn1, StringComparison.Ordinal);
        Assert.StartsWith(renderedTurn1 + answer, renderedTurn2, StringComparison.Ordinal);
    }

    [Fact]
    public void ParserAcceptsDocumentedChatCompletionsJsonSchemaShape()
    {
        using var body = JsonDocument.Parse("""
        {
          "response_format": {
            "type": "json_schema",
            "json_schema": {
              "name": "research_paper_extraction",
              "strict": true,
              "schema": {
                "type": "object",
                "properties": {
                  "title": { "type": "string" }
                },
                "required": ["title"],
                "additionalProperties": false
              }
            }
          }
        }
        """);

        bool ok = OpenAIResponseFormatParser.TryParse(body.RootElement, out var format, out var error);

        Assert.True(ok);
        Assert.Null(error);
        Assert.NotNull(format);
        Assert.Equal(StructuredOutputKind.JsonSchema, format!.Kind);
        Assert.Equal("research_paper_extraction", format.Name);
        Assert.True(format.Strict);
    }

    [Fact]
    public void JsonSchemaValidationRejectsRootAnyOfAndMissingRequired()
    {
        var format = StructuredOutputFormat.JsonSchema("bad_schema", """
        {
          "anyOf": [
            {
              "type": "object",
              "properties": {
                "answer": { "type": "string" }
              },
              "required": [],
              "additionalProperties": false
            }
          ]
        }
        """);

        var validation = StructuredOutputValidator.ValidateSchema(format);

        Assert.False(validation.IsValid);
        Assert.Contains(validation.Errors, e => e.Contains("root schema", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(validation.Errors, e => e.Contains("required", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void JsonSchemaValidationRequiresStrictAndAdditionalPropertiesFalse()
    {
        var format = StructuredOutputFormat.JsonSchema("weather", """
        {
          "type": "object",
          "properties": {
            "city": { "type": "string" }
          },
          "required": ["city"]
        }
        """, strict: false);

        var validation = StructuredOutputValidator.ValidateSchema(format);

        Assert.False(validation.IsValid);
        Assert.Contains(validation.Errors, e => e.Contains("strict", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(validation.Errors, e => e.Contains("additionalProperties", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void JsonObjectNormalizationExtractsCodeFencedJson()
    {
        var normalized = StructuredOutputValidator.NormalizeOutput("""
        Here you go:

        ```json
        {
          "answer": 5
        }
        ```
        """, StructuredOutputFormat.JsonObject());

        Assert.True(normalized.IsValid, normalized.ErrorMessage);
        Assert.Equal("""{"answer":5}""", normalized.NormalizedContent);
    }

    [Fact]
    public void JsonSchemaNormalizationDropsExtrasFillsNullableFieldsAndPreservesSchemaOrder()
    {
        var format = StructuredOutputFormat.JsonSchema("result", """
        {
          "type": "object",
          "properties": {
            "answer": { "type": "string" },
            "optional_note": { "type": ["string", "null"] },
            "done": { "type": "boolean" }
          },
          "required": ["answer", "optional_note", "done"],
          "additionalProperties": false
        }
        """);

        var normalized = StructuredOutputValidator.NormalizeOutput("""
        {
          "done": true,
          "extra": "remove me",
          "answer": "ok"
        }
        """, format);

        Assert.True(normalized.IsValid, normalized.ErrorMessage);
        Assert.Equal("""{"answer":"ok","optional_note":null,"done":true}""", normalized.NormalizedContent);
    }

    [Fact]
    public void JsonObjectNormalizationRepairsMalformedJson()
    {
        var normalized = StructuredOutputValidator.NormalizeOutput(
            "Sure! {'answer': 'forty-two', count: 42, valid: True,}",
            StructuredOutputFormat.JsonObject());

        Assert.True(normalized.IsValid, normalized.ErrorMessage);
        using var doc = JsonDocument.Parse(normalized.NormalizedContent);
        Assert.Equal("forty-two", doc.RootElement.GetProperty("answer").GetString());
        Assert.Equal(42, doc.RootElement.GetProperty("count").GetInt32());
        Assert.True(doc.RootElement.GetProperty("valid").GetBoolean());
    }

    [Fact]
    public void JsonObjectNormalizationWrapsPlainTextAsJson()
    {
        var normalized = StructuredOutputValidator.NormalizeOutput(
            "Sorry, I can only answer in prose.",
            StructuredOutputFormat.JsonObject());

        Assert.True(normalized.IsValid, normalized.ErrorMessage);
        using var doc = JsonDocument.Parse(normalized.NormalizedContent);
        Assert.Equal("Sorry, I can only answer in prose.",
            doc.RootElement.GetProperty("response").GetString());
    }

    [Fact]
    public void JsonSchemaNormalizationRepairsTruncatedOutput()
    {
        var format = StructuredOutputFormat.JsonSchema("result", """
        {
          "type": "object",
          "properties": {
            "answer": { "type": "string" },
            "optional_note": { "type": ["string", "null"] },
            "done": { "type": "boolean" }
          },
          "required": ["answer", "optional_note", "done"],
          "additionalProperties": false
        }
        """);

        // output cut off by max_tokens before the closing brace
        var normalized = StructuredOutputValidator.NormalizeOutput(
            "{\"done\": true, \"answer\": \"ok\"", format);

        Assert.True(normalized.IsValid, normalized.ErrorMessage);
        Assert.Equal("""{"answer":"ok","optional_note":null,"done":true}""", normalized.NormalizedContent);
    }

    [Fact]
    public void JsonSchemaNormalizationSupportsDefsAndAnyOf()
    {
        var format = StructuredOutputFormat.JsonSchema("container", """
        {
          "type": "object",
          "properties": {
            "item": {
              "anyOf": [
                { "$ref": "#/$defs/person" },
                {
                  "type": "object",
                  "properties": {
                    "city": { "type": "string" }
                  },
                  "required": ["city"],
                  "additionalProperties": false
                }
              ]
            }
          },
          "$defs": {
            "person": {
              "type": "object",
              "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
              },
              "required": ["name", "age"],
              "additionalProperties": false
            }
          },
          "required": ["item"],
          "additionalProperties": false
        }
        """);

        var normalized = StructuredOutputValidator.NormalizeOutput("""
        {
          "item": {
            "age": 30,
            "name": "Ada",
            "ignored": true
          }
        }
        """, format);

        Assert.True(normalized.IsValid, normalized.ErrorMessage);
        Assert.Equal("""{"item":{"name":"Ada","age":30}}""", normalized.NormalizedContent);
    }
}


