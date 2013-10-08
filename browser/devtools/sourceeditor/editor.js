/* vim:set ts=2 sw=2 sts=2 et tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

"use strict";

const { Cu, Cc, Ci, components } = require("chrome");

const TAB_SIZE    = "devtools.editor.tabsize";
const EXPAND_TAB  = "devtools.editor.expandtab";
const L10N_BUNDLE = "chrome://browser/locale/devtools/sourceeditor.properties";
const XUL_NS      = "http://www.mozilla.org/keymaster/gatekeeper/there.is.only.xul";

const promise = require("sdk/core/promise");
const events  = require("devtools/shared/event-emitter");

Cu.import("resource://gre/modules/Services.jsm");
const L10N = Services.strings.createBundle(L10N_BUNDLE);

// CM_STYLES, CM_SCRIPTS and CM_IFRAME represent the HTML,
// JavaScript and CSS that is injected into an iframe in
// order to initialize a CodeMirror instance.

const CM_STYLES   = [
  "chrome://browser/content/devtools/codemirror/codemirror.css",
  "chrome://browser/content/devtools/codemirror/dialog.css"
];

const CM_SCRIPTS  = [
  "chrome://browser/content/devtools/codemirror/codemirror.js",
  "chrome://browser/content/devtools/codemirror/dialog.js",
  "chrome://browser/content/devtools/codemirror/searchcursor.js",
  "chrome://browser/content/devtools/codemirror/search.js",
  "chrome://browser/content/devtools/codemirror/matchbrackets.js",
  "chrome://browser/content/devtools/codemirror/comment.js"
];

const CM_IFRAME   =
  "data:text/html;charset=utf8,<!DOCTYPE html>" +
  "<html dir='ltr'>" +
  "  <head>" +
  "    <style>" +
  "      html, body { height: 100%; }" +
  "      body { margin: 0; overflow: hidden; }" +
  "      .CodeMirror { width: 100%; height: 100% !important; }" +
  "    </style>" +
[ "    <link rel='stylesheet' href='" + style + "'>" for (style of CM_STYLES) ].join("\n") +
  "  </head>" +
  "  <body></body>" +
  "</html>";

const CM_MAPPING = [
  "focus",
  "hasFocus",
  "setCursor",
  "getCursor",
  "somethingSelected",
  "setSelection",
  "getSelection",
  "replaceSelection",
  "undo",
  "redo",
  "clearHistory",
  "posFromIndex",
  "openDialog"
];

const CM_JUMP_DIALOG = [
  L10N.GetStringFromName("gotoLineCmd.promptTitle")
    + " <input type=text style='width: 10em'/>"
];

const editors = new WeakMap();

Editor.modes = {
  text: { name: "text" },
  js:   { name: "javascript", url: "chrome://browser/content/devtools/codemirror/javascript.js" }
};

function ctrl(k) {
  return (Services.appinfo.OS == "Darwin" ? "Cmd-" : "Ctrl-") + k;
}

/**
 * A very thin wrapper around CodeMirror. Provides a number
 * of helper methods to make our use of CodeMirror easier and
 * another method, appendTo, to actually create and append
 * the CodeMirror instance.
 *
 * Note that Editor doesn't expose CodeMirror instance to the
 * outside world.
 *
 * Constructor accepts one argument, config. It is very
 * similar to the CodeMirror configuration object so for most
 * properties go to CodeMirror's documentation (see below).
 *
 * Other than that, it accepts one additional and optional
 * property contextMenu. This property should be an ID of
 * an element we can use as a context menu.
 *
 * This object is also an event emitter.
 *
 * CodeMirror docs: http://codemirror.net/doc/manual.html
 */
function Editor(config) {
  const tabSize = Services.prefs.getIntPref(TAB_SIZE);
  const useTabs = !Services.prefs.getBoolPref(EXPAND_TAB);

  this.version = null;
  this.config = {
    value:          "",
    mode:           Editor.modes.text,
    indentUnit:     tabSize,
    tabSize:        tabSize,
    contextMenu:    null,
    matchBrackets:  true,
    extraKeys:      {},
    indentWithTabs: useTabs,
  };

  // Overwrite default config with user-provided, if needed.
  Object.keys(config).forEach((k) => this.config[k] = config[k]);

  // Additional shortcuts.
  this.config.extraKeys[ctrl("J")] = (cm) => this.jumpToLine();
  this.config.extraKeys[ctrl("/")] = "toggleComment";

  // Disable ctrl-[ and ctrl-] because toolbox uses those
  // shortcuts.
  this.config.extraKeys[ctrl("[")] = false;
  this.config.extraKeys[ctrl("]")] = false;

  // Overwrite default tab behavior. If something is selected,
  // indent those lines. If nothing is selected and we're
  // indenting with tabs, insert one tab. Otherwise insert N
  // whitespaces where N == indentUnit option.
  this.config.extraKeys.Tab = (cm) => {
    if (cm.somethingSelected())
      return void cm.indentSelection("add");

    if (this.config.indentWithTabs)
      return void cm.replaceSelection("\t", "end", "+input");

    var num = cm.getOption("indentUnit");
    if (cm.getCursor().ch !== 0) num -= 1;
    cm.replaceSelection(" ".repeat(num), "end", "+input");
  };

  events.decorate(this);
}

Editor.prototype = {
  version: null,
  config:  null,

  /**
   * Appends the current Editor instance to the element specified by
   * the only argument 'el'. This method actually creates and loads
   * CodeMirror and all its dependencies.
   *
   * This method is asynchronous and returns a promise.
   */
  appendTo: function (el) {
    let def = promise.defer();
    let cm  = editors.get(this);
    let doc = el.ownerDocument;
    let env = doc.createElementNS(XUL_NS, "iframe");
    env.flex = 1;

    if (cm)
      throw new Error("You can append an editor only once.");

    let onLoad = () => {
      // Once the iframe is loaded, we can inject CodeMirror
      // and its dependencies into its DOM.
      env.removeEventListener("load", onLoad, true);
      let win = env.contentWindow.wrappedJSObject;

      CM_SCRIPTS.forEach((url) =>
        Services.scriptloader.loadSubScript(url, win, "utf8"));

      // Plain text mode doesn't need any additional files,
      // all other modes (js, html, etc.) do.
      if (this.config.mode.name !== "text")
        Services.scriptloader.loadSubScript(this.config.mode.url, win, "utf8");

      // Create a CodeMirror instance add support for context menus and
      // overwrite the default controller (otherwise items in the top and
      // context menus won't work).

      cm = win.CodeMirror(win.document.body, this.config);
      cm.getWrapperElement().addEventListener("contextmenu", (ev) => {
        ev.preventDefault();
        this.showContextMenu(doc, ev.screenX, ev.screenY);
      }, false);

      cm.on("change", () => this.emit("change"));
      doc.defaultView.controllers.insertControllerAt(0, controller(this, doc.defaultView));

      editors.set(this, cm);
      def.resolve();
    };

    env.addEventListener("load", onLoad, true);
    env.setAttribute("src", CM_IFRAME);
    el.appendChild(env);

    this.once("destroy", () => el.removeChild(env));
    return def.promise;
  },

  /**
   * Returns true if there's something to undo and false otherwise.
   */
  canUndo: function () {
    let cm = editors.get(this);
    return cm.historySize().undo > 0;
  },

  /**
   * Returns true if there's something to redo and false otherwise.
   */
  canRedo: function () {
    let cm = editors.get(this);
    return cm.historySize().redo > 0;
  },

  /**
   * Calculates and returns one or more {line, ch} objects for
   * a zero-based index who's value is relative to the start of
   * the editor's text.
   *
   * If only one argument is given, this method returns a single
   * {line,ch} object. Otherwise it returns an array.
   */
  getPosition: function (...args) {
    let cm = editors.get(this);
    let res = args.map((ind) => cm.posFromIndex(ind));
    return args.length === 1 ? res[0] : res;
  },

  /**
   * The reverse of getPosition. Similarly to getPosition this
   * method returns a single value if only one argument was given
   * and an array otherwise.
   */
  getOffset: function (...args) {
    let cm = editors.get(this);
    let res = args.map((pos) => cm.indexFromPos(pos));
    return args.length > 1 ? res : res[0];
  },

  /**
   * Returns text from the text area.
   */
  getText: function () {
    let cm = editors.get(this);
    return cm.getValue();
  },

  /**
   * Replaces whatever is in the text area with the contents of
   * the 'value' argument.
   */
  setText: function (value) {
    let cm = editors.get(this);
    cm.setValue(value);
  },

  /**
   * Replaces contents of a text area within the from/to {line, ch}
   * range. If neither from nor to arguments are provided works
   * exactly like setText. If only from object is provided, inserts
   * text at that point.
   */
  replaceText: function (value, from, to) {
    let cm = editors.get(this);

    if (!from)
      return void this.setText(value);

    if (!to) {
      let text = cm.getRange({ line: 0, ch: 0 }, from);
      return void this.setText(text + value);
    }

    cm.replaceRange(value, from, to);
  },

  /**
   * Deselects contents of the text area.
   */
  dropSelection: function () {
    if (!this.somethingSelected())
      return;

    this.setCursor(this.getCursor());
  },

  /**
   * Marks the contents as clean and returns the current
   * version number.
   */
  markClean: function () {
    let cm = editors.get(this);
    this.version = cm.changeGeneration();
    return this.version;
  },

  /**
   * Returns true if contents of the text area are
   * clean i.e. no changes were made since the last version.
   */
  isClean: function () {
    let cm = editors.get(this);
    return cm.isClean(this.version);
  },

  /**
   * Displays a context menu at the point x:y. The first
   * argument, container, should be a DOM node that contains
   * a context menu element specified by the ID from
   * config.contextMenu.
   */
  showContextMenu: function (container, x, y) {
    if (this.config.contextMenu == null)
      return;

    let popup = container.getElementById(this.config.contextMenu);
    popup.openPopupAtScreen(x, y, true);
  },

  /**
   * This method opens an in-editor dialog asking for a line to
   * jump to. Once given, it changes cursor to that line.
   */
  jumpToLine: function () {
    this.openDialog(CM_JUMP_DIALOG, (line) =>
      this.setCursor({ line: line - 1, ch: 0 }));
  },

  destroy: function () {
    this.config  = null;
    this.version = null;
    this.emit("destroy");
  }
};

// Since Editor is a thin layer over CodeMirror some methods
// are mapped directly—without any changes.

CM_MAPPING.forEach(function (name) {
  Editor.prototype[name] = function (...args) {
    let cm = editors.get(this);
    return cm[name].apply(cm, args);
  };
});

/**
 * Returns a controller object that can be used for
 * editor-specific commands such as find, jump to line,
 * copy/paste, etc.
 */
function controller(ed, view) {
  return {
    supportsCommand: function (cmd) {
      let cm = editors.get(ed);

      switch (cmd) {
        case "cmd_find":
        case "cmd_findAgain":
        case "cmd_findPrevious":
        case "cmd_gotoLine":
        case "cmd_undo":
        case "cmd_redo":
        case "cmd_cut":
        case "cmd_paste":
        case "cmd_delete":
        case "cmd_selectAll":
          return true;
      }

      return false;
    },

    isCommandEnabled: function (cmd) {
      let cm = editors.get(ed);

      switch (cmd) {
        case "cmd_find":
        case "cmd_gotoLine":
        case "cmd_selectAll":
          return true;
        case "cmd_findAgain":
          return cm.state.search != null && cm.state.search.query != null;
        case "cmd_undo":
          return ed.canUndo();
        case "cmd_redo":
          return ed.canRedo();
        case "cmd_cut":
          return cm.getOption("readOnly") !== true && ed.somethingSelected();
        case "cmd_delete":
          return ed.somethingSelected();
        case "cmd_paste":
          return cm.getOption("readOnly") !== true;
      }

      return false;
    },

    doCommand: function (cmd) {
      let cm  = editors.get(ed);
      let map = {
        "cmd_selectAll": "selectAll",
        "cmd_find": "find",
        "cmd_undo": "undo",
        "cmd_redo": "redo",
        "cmd_delete": "delCharAfter",
        "cmd_findAgain": "findNext"
      };

      if (map[cmd])
        return void cm.execCommand(map[cmd]);

      if (cmd === "cmd_cut")
        return void view.goDoCommand("cmd_cut");

      if (cmd === "cmdste")
        return void view.goDoCommand("cmd_paste");

      if (cmd == "cmd_gotoLine")
        ed.jumpToLine(cm);
    },

    onEvent: function () {}
  };
}

module.exports = Editor;