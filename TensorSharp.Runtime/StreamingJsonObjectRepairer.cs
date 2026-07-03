// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Incrementally extracts and repairs the first JSON object in a model's
    /// output stream so that the concatenation of everything returned by
    /// <see cref="Feed"/> plus <see cref="Finish"/> is always exactly one JSON
    /// object that a strict JSON parser accepts.
    ///
    /// Beyond the brace balancing the old streaming filter did (drop markdown
    /// fences / prose before the opening <c>{</c>, drop trailing tags after the
    /// matching <c>}</c>, keep braces inside strings intact), this repairs the
    /// malformed JSON local models commonly emit, on the fly:
    /// <list type="bullet">
    ///   <item>truncated output (max_tokens hit mid-object): <see cref="Finish"/>
    ///     closes open strings, completes dangling keys/colons with <c>null</c>,
    ///     completes partial <c>true/false/null</c> literals and numbers, and
    ///     closes every open object/array;</item>
    ///   <item>single-quoted strings and unquoted object keys are rewritten to
    ///     double-quoted JSON strings;</item>
    ///   <item>Python-style literals (<c>True</c>, <c>False</c>, <c>None</c>,
    ///     <c>NaN</c>, <c>Infinity</c>) become <c>true</c>/<c>false</c>/<c>null</c>;</item>
    ///   <item>trailing commas are held back and dropped when a closer follows;
    ///     missing commas between elements are inserted;</item>
    ///   <item>raw control characters inside strings (e.g. real newlines) are
    ///     escaped; invalid escape sequences are neutralized;</item>
    ///   <item>invalid numbers (<c>+5</c>, <c>012</c>, <c>1.</c>, <c>1e</c>) are
    ///     normalized to valid JSON numbers;</item>
    ///   <item>if the stream contains no JSON object at all, <see cref="Finish"/>
    ///     falls back to <c>{"response": "&lt;raw text&gt;"}</c> (or <c>{}</c> for
    ///     an empty stream) so the client still receives parseable JSON.</item>
    /// </list>
    ///
    /// Valid JSON passes through byte-for-byte unchanged (including inter-token
    /// whitespace), so well-behaved model output is streamed exactly as produced.
    /// </summary>
    public sealed class StreamingJsonObjectRepairer
    {
        private enum Container { Object, Array }

        private enum Expect
        {
            KeyOrEnd,   // inside an object, before a key (after '{' or a comma)
            Colon,      // inside an object, after a key
            Value,      // inside an object, after ':'
            ValueOrEnd, // inside an array, before a value (after '[' or a comma)
            CommaOrEnd, // after a complete value
        }

        // Text retained before the first '{' is capped so a runaway stream that
        // never starts an object cannot grow memory unboundedly.
        private const int MaxProseChars = 1 << 20;

        private static readonly Regex ValidJsonNumber =
            new("^-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$", RegexOptions.Compiled);

        private readonly List<Container> _stack = new();
        private readonly StringBuilder _prose = new();   // text seen before the object starts
        private readonly StringBuilder _pending = new(); // held separator comma (+ following whitespace)
        private readonly StringBuilder _token = new();   // in-flight number or bare-word token
        private readonly StringBuilder _escape = new();  // in-flight string escape sequence

        private bool _started;
        private bool _done;
        private Expect _expect;
        private bool _inString;
        private bool _stringIsKey;
        private char _stringDelim;
        private bool _tokenIsNumber;
        private bool _tokenIsKey;

        /// <summary>True once the opening '{' of the object has been seen.</summary>
        public bool Started => _started;

        /// <summary>True once the complete JSON object has been emitted.</summary>
        public bool Done => _done;

        /// <summary>
        /// Feed a raw output fragment; returns the (possibly repaired) portion
        /// that belongs to the JSON object and should be streamed to the client.
        /// </summary>
        public string Feed(string text)
        {
            if (_done || string.IsNullOrEmpty(text))
                return string.Empty;

            var sb = new StringBuilder(text.Length);
            foreach (char ch in text)
            {
                if (_done)
                    break;
                ProcessChar(ch, sb);
            }
            return sb.ToString();
        }

        /// <summary>
        /// Signal end-of-stream; returns the suffix that must be appended to make
        /// everything emitted so far exactly one valid JSON object. If no object
        /// ever started, returns a whole fallback object (<c>{}</c> or
        /// <c>{"response": "&lt;raw text&gt;"}</c>). Idempotent: later calls (and
        /// later <see cref="Feed"/> calls) return an empty string.
        /// </summary>
        public string Finish()
        {
            if (_done)
                return string.Empty;

            _done = true;

            if (!_started)
            {
                string prose = _prose.ToString().Trim();
                if (prose.Length == 0)
                    return "{}";
                return JsonSerializer.Serialize(new Dictionary<string, string> { ["response"] = prose });
            }

            var sb = new StringBuilder();

            if (_inString)
            {
                _escape.Clear(); // drop a dangling, incomplete escape sequence
                sb.Append('"');
                _inString = false;
                _expect = _stringIsKey ? Expect.Colon : Expect.CommaOrEnd;
            }

            if (_token.Length > 0)
                CommitToken(sb, atEndOfStream: true);

            _pending.Clear(); // a trailing separator comma is dropped

            if (_expect == Expect.Colon)
                sb.Append(":null");
            else if (_expect == Expect.Value)
                sb.Append("null");

            for (int i = _stack.Count - 1; i >= 0; i--)
                sb.Append(_stack[i] == Container.Object ? '}' : ']');
            _stack.Clear();

            return sb.ToString();
        }

        // ---- Character processing ---------------------------------------------

        private void ProcessChar(char ch, StringBuilder sb)
        {
            if (!_started)
            {
                if (ch == '{')
                {
                    _started = true;
                    _stack.Add(Container.Object);
                    _expect = Expect.KeyOrEnd;
                    sb.Append('{');
                }
                else if (_prose.Length < MaxProseChars)
                {
                    _prose.Append(ch);
                }
                return;
            }

            if (_inString)
            {
                ProcessStringChar(ch, sb);
                return;
            }

            if (_token.Length > 0)
            {
                if (_tokenIsNumber ? IsNumberChar(ch) : IsWordChar(ch))
                {
                    _token.Append(ch);
                    return;
                }
                CommitToken(sb, atEndOfStream: false);
                // fall through: ch still needs structural handling
            }

            ProcessStructuralChar(ch, sb);
        }

        private void ProcessStructuralChar(char ch, StringBuilder sb)
        {
            if (char.IsWhiteSpace(ch))
            {
                // Whitespace after a held comma stays held so the pair is emitted
                // (or dropped) in original order; elsewhere it passes through so
                // valid pretty-printed JSON streams unchanged.
                if (_pending.Length > 0)
                    _pending.Append(ch);
                else
                    sb.Append(ch);
                return;
            }

            switch (_expect)
            {
                case Expect.KeyOrEnd:
                    if (ch == '"' || ch == '\'')
                    {
                        FlushPending(sb);
                        BeginString(ch, isKey: true, sb);
                    }
                    else if (ch == '}')
                        CloseContainer(Container.Object, sb);
                    else if (ch == ']')
                        CloseContainer(Container.Array, sb);
                    else if (IsWordStartChar(ch))
                        BeginToken(ch, isNumber: false, isKey: true);
                    // stray commas or other garbage before a key are skipped
                    break;

                case Expect.Colon:
                    if (ch == ':' || ch == '=')
                    {
                        sb.Append(':');
                        _expect = Expect.Value;
                    }
                    else if (ch == '}' || ch == ']')
                    {
                        Container type = ch == '}' ? Container.Object : Container.Array;
                        if (_stack.Contains(type)) // stray closers are skipped un-patched
                        {
                            sb.Append(":null");
                            CloseContainer(type, sb);
                        }
                    }
                    else if (ch == ',')
                    {
                        sb.Append(":null");
                        HoldComma();
                    }
                    else if (IsValueStartChar(ch))
                    {
                        // missing colon between key and value — insert one
                        sb.Append(':');
                        _expect = Expect.Value;
                        ProcessStructuralChar(ch, sb);
                    }
                    // other garbage between key and colon is skipped
                    break;

                case Expect.Value:
                case Expect.ValueOrEnd:
                    if (ch == '"' || ch == '\'')
                    {
                        FlushPending(sb);
                        BeginString(ch, isKey: false, sb);
                    }
                    else if (ch == '{')
                    {
                        FlushPending(sb);
                        sb.Append('{');
                        _stack.Add(Container.Object);
                        _expect = Expect.KeyOrEnd;
                    }
                    else if (ch == '[')
                    {
                        FlushPending(sb);
                        sb.Append('[');
                        _stack.Add(Container.Array);
                        _expect = Expect.ValueOrEnd;
                    }
                    else if (ch == '}' || ch == ']')
                    {
                        Container type = ch == '}' ? Container.Object : Container.Array;
                        if (_stack.Contains(type)) // stray closers are skipped un-patched
                        {
                            if (_expect == Expect.Value)
                            {
                                FlushPending(sb);
                                sb.Append("null"); // dangling ':' before the closer
                            }
                            CloseContainer(type, sb);
                        }
                    }
                    else if (ch == ',')
                    {
                        // missing value before a comma — substitute null
                        FlushPending(sb);
                        sb.Append("null");
                        HoldComma();
                    }
                    else if (IsNumberStartChar(ch))
                        BeginToken(ch, isNumber: true, isKey: false);
                    else if (IsWordStartChar(ch))
                        BeginToken(ch, isNumber: false, isKey: false);
                    // other garbage where a value should start is skipped
                    break;

                case Expect.CommaOrEnd:
                    if (ch == ',')
                        HoldComma();
                    else if (ch == '}')
                        CloseContainer(Container.Object, sb);
                    else if (ch == ']')
                        CloseContainer(Container.Array, sb);
                    else if (ch == '"' || ch == '\'')
                    {
                        // missing comma before the next key/element — insert one
                        HoldComma();
                        ProcessStructuralChar(ch, sb);
                    }
                    else if ((ch == '{' || ch == '[' || IsNumberStartChar(ch)) &&
                             Top() == Container.Array)
                    {
                        // missing comma between array elements — insert one
                        HoldComma();
                        ProcessStructuralChar(ch, sb);
                    }
                    // other trailing garbage after a value is skipped
                    break;
            }
        }

        private void ProcessStringChar(char ch, StringBuilder sb)
        {
            if (_escape.Length > 0)
            {
                if (_escape.Length == 1) // pending "\"
                {
                    switch (ch)
                    {
                        case '"':
                        case '\\':
                        case '/':
                        case 'b':
                        case 'f':
                        case 'n':
                        case 'r':
                        case 't':
                            sb.Append('\\').Append(ch);
                            _escape.Clear();
                            return;
                        case 'u':
                            _escape.Append('u');
                            return;
                        case '\'':
                            // \' from a single-quoted string — a plain quote in JSON
                            sb.Append('\'');
                            _escape.Clear();
                            return;
                        default:
                            // invalid escape — keep the backslash as literal text
                            sb.Append("\\\\");
                            _escape.Clear();
                            ProcessStringChar(ch, sb);
                            return;
                    }
                }

                // pending "\u" + collected hex digits
                if (Uri.IsHexDigit(ch))
                {
                    _escape.Append(ch);
                    if (_escape.Length == 6)
                    {
                        sb.Append(_escape);
                        _escape.Clear();
                    }
                    return;
                }

                // invalid \u escape — emit it as literal text and reprocess ch
                sb.Append("\\\\").Append(_escape, 1, _escape.Length - 1);
                _escape.Clear();
                ProcessStringChar(ch, sb);
                return;
            }

            if (ch == '\\')
            {
                _escape.Append('\\');
                return;
            }

            if (ch == _stringDelim)
            {
                sb.Append('"');
                _inString = false;
                _expect = _stringIsKey ? Expect.Colon : Expect.CommaOrEnd;
                return;
            }

            if (ch == '"')
            {
                // a double quote inside a single-quoted string must be escaped
                sb.Append("\\\"");
                return;
            }

            if (ch < 0x20)
            {
                switch (ch)
                {
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        sb.Append("\\u").Append(((int)ch).ToString("x4", CultureInfo.InvariantCulture));
                        break;
                }
                return;
            }

            sb.Append(ch);
        }

        // ---- Tokens -------------------------------------------------------------

        private void BeginToken(char ch, bool isNumber, bool isKey)
        {
            _token.Append(ch);
            _tokenIsNumber = isNumber;
            _tokenIsKey = isKey;
        }

        private void CommitToken(StringBuilder sb, bool atEndOfStream)
        {
            string token = _token.ToString();
            _token.Clear();

            if (_tokenIsNumber)
            {
                FlushPending(sb);
                sb.Append(FixNumber(token));
                _expect = Expect.CommaOrEnd;
                return;
            }

            if (_tokenIsKey)
            {
                FlushPending(sb);
                sb.Append('"').Append(token).Append('"');
                _expect = Expect.Colon;
                return;
            }

            string mapped = MapWordValue(token, allowPrefix: atEndOfStream);
            if (mapped == null)
                return; // unrecognized bare word where a value belongs — skip it

            FlushPending(sb);
            sb.Append(mapped);
            _expect = Expect.CommaOrEnd;
        }

        private static string MapWordValue(string token, bool allowPrefix)
        {
            switch (token)
            {
                case "true":
                case "True":
                case "TRUE":
                    return "true";
                case "false":
                case "False":
                case "FALSE":
                    return "false";
                case "null":
                case "Null":
                case "NULL":
                case "None":
                case "nil":
                case "undefined":
                case "NaN":
                case "Infinity":
                    return "null";
            }

            if (allowPrefix && token.Length > 0)
            {
                // complete a literal that was cut off by end-of-stream
                string lower = token.ToLowerInvariant();
                if ("true".StartsWith(lower, StringComparison.Ordinal))
                    return "true";
                if ("false".StartsWith(lower, StringComparison.Ordinal))
                    return "false";
                if ("null".StartsWith(lower, StringComparison.Ordinal) ||
                    "none".StartsWith(lower, StringComparison.Ordinal))
                    return "null";
            }

            return null;
        }

        private static string FixNumber(string token)
        {
            string t = token;
            if (t.StartsWith("+", StringComparison.Ordinal))
                t = t.Substring(1);
            if (t.Length == 0)
                return "0";
            if (t.EndsWith(".", StringComparison.Ordinal) ||
                t.EndsWith("e", StringComparison.OrdinalIgnoreCase) ||
                t.EndsWith("+", StringComparison.Ordinal) ||
                t.EndsWith("-", StringComparison.Ordinal))
            {
                t += "0";
            }

            // strip redundant leading zeros ("012" -> "12", "-007" -> "-7")
            int sign = t.StartsWith("-", StringComparison.Ordinal) ? 1 : 0;
            int firstSignificant = sign;
            while (firstSignificant < t.Length - 1 && t[firstSignificant] == '0' &&
                   char.IsDigit(t[firstSignificant + 1]))
            {
                firstSignificant++;
            }
            if (firstSignificant > sign)
                t = t.Substring(0, sign) + t.Substring(firstSignificant);

            if (ValidJsonNumber.IsMatch(t))
                return t;

            return double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out double d)
                   && !double.IsNaN(d) && !double.IsInfinity(d)
                ? d.ToString("R", CultureInfo.InvariantCulture)
                : "0";
        }

        // ---- Structure helpers ---------------------------------------------------

        private void BeginString(char delim, bool isKey, StringBuilder sb)
        {
            _inString = true;
            _stringIsKey = isKey;
            _stringDelim = delim;
            sb.Append('"');
        }

        private void HoldComma()
        {
            _pending.Clear();
            _pending.Append(',');
            _expect = Top() == Container.Object ? Expect.KeyOrEnd : Expect.ValueOrEnd;
        }

        private void FlushPending(StringBuilder sb)
        {
            if (_pending.Length == 0)
                return;
            sb.Append(_pending);
            _pending.Clear();
        }

        private void CloseContainer(Container type, StringBuilder sb)
        {
            if (!_stack.Contains(type))
                return; // stray closer with no matching opener — skip it

            _pending.Clear(); // a separator comma directly before a closer is dropped

            while (_stack.Count > 0)
            {
                Container top = _stack[^1];
                _stack.RemoveAt(_stack.Count - 1);
                sb.Append(top == Container.Object ? '}' : ']');
                if (top == type)
                    break;
            }

            if (_stack.Count == 0)
                _done = true;
            else
                _expect = Expect.CommaOrEnd;
        }

        private Container Top() => _stack.Count > 0 ? _stack[^1] : Container.Object;

        private static bool IsNumberStartChar(char ch)
            => (ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.';

        private static bool IsNumberChar(char ch)
            => (ch >= '0' && ch <= '9') || ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-';

        private static bool IsWordStartChar(char ch)
            => char.IsLetter(ch) || ch == '_' || ch == '$';

        private static bool IsWordChar(char ch)
            => char.IsLetterOrDigit(ch) || ch == '_' || ch == '$';

        private bool IsValueStartChar(char ch)
            => ch == '"' || ch == '\'' || ch == '{' || ch == '[' ||
               IsNumberStartChar(ch) || IsWordStartChar(ch);
    }
}
