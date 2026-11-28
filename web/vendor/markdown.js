/* Markdown + LaTeX renderer for dlqq-llm WebUI.
 * Uses the locally vendored marked, DOMPurify and KaTeX.
 * Model output is untrusted: Markdown HTML is sanitized before insertion,
 * and math placeholders are replaced with KaTeX output only after that.
 */
(function () {
  "use strict";

  var MATH = [];
  var MARK_RE = /@@DLQQ_MATH_(\d+)@@/g;

  function placeholder(tex, display) {
    var i = MATH.push({ tex: tex, display: !!display }) - 1;
    return "@@DLQQ_MATH_" + i + "@@";
  }

  function mathStart(src, block) {
    var re = block ? /\$\$|\\\[/ : /\$|\\\(/;
    var m = re.exec(src);
    return m ? m.index : undefined;
  }

  var blockMath = {
    name: "dlqqMathBlock",
    level: "block",
    start: function (src) { return mathStart(src, true); },
    tokenizer: function (src) {
      var m = /^\$\$([\s\S]+?)\$\$(?:\n+|$)/.exec(src);
      if (m) return { type: "dlqqMathBlock", raw: m[0], text: m[1].trim(), display: true };
      m = /^\\\[([\s\S]+?)\\\](?:\n+|$)/.exec(src);
      if (m) return { type: "dlqqMathBlock", raw: m[0], text: m[1].trim(), display: true };
      return undefined;
    },
    renderer: function (token) { return placeholder(token.text, token.display); }
  };

  var inlineMath = {
    name: "dlqqMathInline",
    level: "inline",
    start: function (src) { return mathStart(src, false); },
    tokenizer: function (src) {
      var m = /^\$([^$\n]+?)\$/.exec(src);
      if (m) return { type: "dlqqMathInline", raw: m[0], text: m[1].trim(), display: false };
      m = /^\\\(([\s\S]+?)\\\)/.exec(src);
      if (m) return { type: "dlqqMathInline", raw: m[0], text: m[1].trim(), display: false };
      return undefined;
    },
    renderer: function (token) { return placeholder(token.text, token.display); }
  };

  var configured = false;
  function configure() {
    if (configured || typeof marked === "undefined") return;
    configured = true;
    marked.use({ extensions: [blockMath, inlineMath] });
  }

  function hydrateMath(root) {
    if (typeof katex === "undefined") return;
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    var nodes = [];
    var node;
    while ((node = walker.nextNode())) {
      if (node.nodeValue && node.nodeValue.indexOf("@@DLQQ_MATH_") >= 0) nodes.push(node);
    }
    nodes.forEach(function (textNode) {
      var parent = textNode.parentElement;
      if (parent && (parent.tagName === "CODE" || parent.tagName === "PRE")) return;

      var text = textNode.nodeValue;
      var frag = document.createDocumentFragment();
      var last = 0;
      var m;
      MARK_RE.lastIndex = 0;
      while ((m = MARK_RE.exec(text))) {
        if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        var item = MATH[Number(m[1])];
        var span = document.createElement("span");
        span.className = item && item.display ? "math-display" : "math-inline";
        if (item) {
          try {
            katex.render(item.tex, span, {
              displayMode: !!item.display,
              throwOnError: false,
              trust: false,
              strict: "ignore"
            });
          } catch (e) {
            span.textContent = item.display ? "$$" + item.tex + "$$" : "$" + item.tex + "$";
          }
        }
        frag.appendChild(span);
        last = m.index + m[0].length;
      }
      if (last === 0) return;
      if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
      textNode.parentNode.replaceChild(frag, textNode);
    });
  }

  function demoteMath(root) {
    // 流式阶段不跑 KaTeX：把占位符还原成 $...$ 字面量，等 done 后再正式渲染。
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    var nodes = [];
    var node;
    while ((node = walker.nextNode())) {
      if (node.nodeValue && node.nodeValue.indexOf("@@DLQQ_MATH_") >= 0) nodes.push(node);
    }
    nodes.forEach(function (textNode) {
      var parent = textNode.parentElement;
      if (parent && (parent.tagName === "CODE" || parent.tagName === "PRE")) return;
      textNode.nodeValue = textNode.nodeValue.replace(MARK_RE, function (m, idx) {
        var item = MATH[Number(idx)];
        if (!item) return "";
        return item.display ? "$$" + item.tex + "$$" : "$" + item.tex + "$";
      });
    });
  }

  function renderMarkdown(target, text, opts) {
    if (!target) return;
    if (typeof marked === "undefined" || typeof DOMPurify === "undefined") {
      target.textContent = text || "";
      return;
    }
    configure();
    MATH.length = 0;
    var raw = String(text || "");
    if (typeof katex === "undefined") {
      target.textContent = raw;
      return;
    }
    var html;
    try {
      html = marked.parse(raw, { gfm: true, breaks: true, async: false });
      html = DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
    } catch (e) {
      target.textContent = raw;
      return;
    }
    target.classList.add("md-body");
    target.innerHTML = html;
    if (opts && opts.streaming) demoteMath(target);
    else hydrateMath(target);
  }

  window.renderMarkdown = renderMarkdown;
})();
