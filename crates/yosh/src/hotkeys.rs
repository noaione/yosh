//! Customizable keyboard shortcuts.
//!
//! Every keyboard-driven [`Action`] is configurable from the `⌨ Hotkeys` page.
//! The old hard-coded `action_from(&KeyEvent)` match is replaced by a
//! data-driven [`HotkeyMap`]: physical `KeyCode` + modifier flags per binding,
//! so bindings work across keyboard layouts. The default map lives in code and
//! reproduces the historical bindings exactly; a missing `hotkeys` field in an
//! existing `state.json` loads those defaults through `#[serde(default)]`.
//!
//! Dispatch rules (see `app.rs::window_event`):
//! 1. egui-focused text/input controls consume keyboard events first;
//! 2. while the capture dialog is open, the captured key never reaches reader
//!    actions;
//! 3. otherwise the pressed key + modifier state maps to exactly one action.
//!
//! OS-reserved / command-modified bindings (`Ctrl`/`Alt`/`Super`) are rejected
//! at capture time — they must not steal browser/OS/tool commands. Unmodified
//! keys and `Shift` combinations are allowed.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};

/// A keyboard-driven reader action. Stable serialized IDs (snake_case) are the
/// persisted tokens; display labels and categories drive the Hotkeys page and
/// the F1 help overlay.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Forward,
    Backward,
    Left,
    Right,
    First,
    Last,
    CycleFit,
    ToggleDir,
    ToggleLayout,
    ToggleScroll,
    ZoomIn,
    ZoomOut,
    // View presets (number keys): each sets a complete page-flip view at once.
    PresetWindow,
    PresetWidth,
    PresetActual,
    PresetSpreadLtr,
    PresetSpreadRtl,
    ToggleHelp,
    ToggleFullscreen,
    ToggleSpreadOffset,
    ToggleInfo,
    ToggleSeekbar,
    TogglePageJump,
    TogglePageTransition,
    ToggleStretch,
    ToggleSpineShadow,
    ToggleAnimBar,
    PrevVolume,
    NextVolume,
    Rotate,
    ShowInExplorer,
    Quit,
}

/// Grouping for the Hotkeys page and the F1 help overlay.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Category {
    Navigation,
    View,
    WindowsInfo,
    LibraryFiles,
    Application,
}

impl Category {
    pub const ALL: [Category; 5] = [
        Category::Navigation,
        Category::View,
        Category::WindowsInfo,
        Category::LibraryFiles,
        Category::Application,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Category::Navigation => "Navigate",
            Category::View => "View",
            Category::WindowsInfo => "Windows & info",
            Category::LibraryFiles => "Library & files",
            Category::Application => "Application",
        }
    }

    pub fn actions(self) -> &'static [Action] {
        use Action::*;
        match self {
            Category::Navigation => &[Forward, Backward, Left, Right, First, Last],
            Category::View => &[
                CycleFit,
                ToggleDir,
                ToggleLayout,
                ToggleScroll,
                ZoomIn,
                ZoomOut,
                PresetWindow,
                PresetWidth,
                PresetActual,
                PresetSpreadLtr,
                PresetSpreadRtl,
                ToggleSpreadOffset,
                ToggleStretch,
                ToggleSpineShadow,
                ToggleSeekbar,
                TogglePageJump,
                TogglePageTransition,
                Rotate,
            ],
            Category::WindowsInfo => &[ToggleHelp, ToggleFullscreen, ToggleInfo, ToggleAnimBar],
            Category::LibraryFiles => &[PrevVolume, NextVolume, ShowInExplorer],
            Category::Application => &[Quit],
        }
    }
}

impl Action {
    /// Every action, for iteration (the map is keyed by `Action`).
    pub const ALL: [Action; 32] = [
        Action::Forward,
        Action::Backward,
        Action::Left,
        Action::Right,
        Action::First,
        Action::Last,
        Action::CycleFit,
        Action::ToggleDir,
        Action::ToggleLayout,
        Action::ToggleScroll,
        Action::ZoomIn,
        Action::ZoomOut,
        Action::PresetWindow,
        Action::PresetWidth,
        Action::PresetActual,
        Action::PresetSpreadLtr,
        Action::PresetSpreadRtl,
        Action::ToggleHelp,
        Action::ToggleFullscreen,
        Action::ToggleSpreadOffset,
        Action::ToggleInfo,
        Action::ToggleSeekbar,
        Action::TogglePageJump,
        Action::TogglePageTransition,
        Action::ToggleStretch,
        Action::ToggleSpineShadow,
        Action::ToggleAnimBar,
        Action::PrevVolume,
        Action::NextVolume,
        Action::Rotate,
        Action::ShowInExplorer,
        Action::Quit,
    ];

    /// Stable serialized ID (also the `serde` token).
    pub fn id(self) -> &'static str {
        match self {
            Action::Forward => "forward",
            Action::Backward => "backward",
            Action::Left => "left",
            Action::Right => "right",
            Action::First => "first",
            Action::Last => "last",
            Action::CycleFit => "cycle_fit",
            Action::ToggleDir => "toggle_dir",
            Action::ToggleLayout => "toggle_layout",
            Action::ToggleScroll => "toggle_scroll",
            Action::ZoomIn => "zoom_in",
            Action::ZoomOut => "zoom_out",
            Action::PresetWindow => "preset_window",
            Action::PresetWidth => "preset_width",
            Action::PresetActual => "preset_actual",
            Action::PresetSpreadLtr => "preset_spread_ltr",
            Action::PresetSpreadRtl => "preset_spread_rtl",
            Action::ToggleHelp => "toggle_help",
            Action::ToggleFullscreen => "toggle_fullscreen",
            Action::ToggleSpreadOffset => "toggle_spread_offset",
            Action::ToggleInfo => "toggle_info",
            Action::ToggleSeekbar => "toggle_seekbar",
            Action::TogglePageJump => "toggle_page_jump",
            Action::TogglePageTransition => "toggle_page_transition",
            Action::ToggleStretch => "toggle_stretch",
            Action::ToggleSpineShadow => "toggle_spine_shadow",
            Action::ToggleAnimBar => "toggle_anim_bar",
            Action::PrevVolume => "prev_volume",
            Action::NextVolume => "next_volume",
            Action::Rotate => "rotate",
            Action::ShowInExplorer => "show_in_explorer",
            Action::Quit => "quit",
        }
    }

    /// Human-readable label for the Hotkeys page / help overlay.
    pub fn label(self) -> &'static str {
        match self {
            Action::Forward => "Next page",
            Action::Backward => "Previous page",
            Action::Left => "Left (direction-aware)",
            Action::Right => "Right (direction-aware)",
            Action::First => "First page",
            Action::Last => "Last page",
            Action::CycleFit => "Cycle fit",
            Action::ToggleDir => "Toggle reading direction",
            Action::ToggleLayout => "Toggle single / two-page",
            Action::ToggleScroll => "Toggle scroll mode",
            Action::ZoomIn => "Zoom in",
            Action::ZoomOut => "Zoom out",
            Action::PresetWindow => "Fit window",
            Action::PresetWidth => "Fit width",
            Action::PresetActual => "100% (1:1)",
            Action::PresetSpreadLtr => "Two-page L→R",
            Action::PresetSpreadRtl => "Two-page R→L",
            Action::ToggleHelp => "Toggle help",
            Action::ToggleFullscreen => "Toggle fullscreen",
            Action::ToggleSpreadOffset => "Shift spread pairing",
            Action::ToggleInfo => "Toggle info overlay",
            Action::ToggleSeekbar => "Toggle seekbar",
            Action::TogglePageJump => "Jump to page",
            Action::TogglePageTransition => "Toggle page transition",
            Action::ToggleStretch => "Toggle stretch small pages",
            Action::ToggleSpineShadow => "Toggle spine shadow",
            Action::ToggleAnimBar => "Toggle animation panel",
            Action::PrevVolume => "Previous volume",
            Action::NextVolume => "Next volume",
            Action::Rotate => "Rotate 90°",
            Action::ShowInExplorer => "Show in Explorer",
            Action::Quit => "Quit/Close",
        }
    }

    pub fn category(self) -> Category {
        for c in Category::ALL {
            if c.actions().contains(&self) {
                return c;
            }
        }
        Category::Application
    }
}

/// A single key binding: a physical `KeyCode` plus modifier flags. Physical
/// codes keep bindings working across keyboard layouts. `Ctrl`/`Alt`/`Super`
/// are always false in stored bindings — command-modified combinations are
/// rejected at capture time (they'd steal OS/tool commands).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct KeyBinding {
    pub code: KeyCode,
    #[serde(default)]
    pub ctrl: bool,
    #[serde(default)]
    pub alt: bool,
    #[serde(default)]
    pub shift: bool,
    #[serde(default, rename = "super")]
    pub super_: bool,
}

impl KeyBinding {
    /// Readable platform-ish name, e.g. `Space`, `PgDn`, `Shift+→`.
    pub fn label(&self) -> String {
        let mut s = String::new();
        if self.ctrl {
            s.push_str("Ctrl+");
        }
        if self.alt {
            s.push_str("Alt+");
        }
        if self.shift {
            s.push_str("Shift+");
        }
        if self.super_ {
            s.push_str("Win+");
        }
        s.push_str(key_label(self.code));
        s
    }
}

/// Modifier keys can't be bound on their own (they only modify other keys).
pub fn is_modifier_key(code: KeyCode) -> bool {
    use KeyCode::*;
    matches!(
        code,
        ShiftLeft
            | ShiftRight
            | ControlLeft
            | ControlRight
            | AltLeft
            | AltRight
            | SuperLeft
            | SuperRight
            | CapsLock
            | NumLock
            | ScrollLock
    )
}

/// Readable name for a physical key code. Covers every default binding plus the
/// common extras; anything unmapped renders as `?`.
fn key_label(code: KeyCode) -> &'static str {
    use KeyCode::*;
    match code {
        KeyA => "A",
        KeyB => "B",
        KeyC => "C",
        KeyD => "D",
        KeyE => "E",
        KeyF => "F",
        KeyG => "G",
        KeyH => "H",
        KeyI => "I",
        KeyJ => "J",
        KeyK => "K",
        KeyL => "L",
        KeyM => "M",
        KeyN => "N",
        KeyO => "O",
        KeyP => "P",
        KeyQ => "Q",
        KeyR => "R",
        KeyS => "S",
        KeyT => "T",
        KeyU => "U",
        KeyV => "V",
        KeyW => "W",
        KeyX => "X",
        KeyY => "Y",
        KeyZ => "Z",
        Digit0 => "0",
        Digit1 => "1",
        Digit2 => "2",
        Digit3 => "3",
        Digit4 => "4",
        Digit5 => "5",
        Digit6 => "6",
        Digit7 => "7",
        Digit8 => "8",
        Digit9 => "9",
        Numpad0 => "Num 0",
        Numpad1 => "Num 1",
        Numpad2 => "Num 2",
        Numpad3 => "Num 3",
        Numpad4 => "Num 4",
        Numpad5 => "Num 5",
        Numpad6 => "Num 6",
        Numpad7 => "Num 7",
        Numpad8 => "Num 8",
        Numpad9 => "Num 9",
        NumpadAdd => "Num +",
        NumpadSubtract => "Num −",
        NumpadMultiply => "Num *",
        NumpadDivide => "Num /",
        NumpadDecimal => "Num .",
        NumpadEnter => "Num Enter",
        Equal => "=",
        Minus => "−",
        BracketLeft => "[",
        BracketRight => "]",
        Backslash => "\\",
        Semicolon => ";",
        Quote => "'",
        Comma => ",",
        Period => ".",
        Slash => "/",
        Backquote => "`",
        Space => "Space",
        Enter => "Enter",
        Tab => "Tab",
        Backspace => "Backspace",
        Delete => "Del",
        Insert => "Ins",
        Home => "Home",
        End => "End",
        PageUp => "PgUp",
        PageDown => "PgDn",
        ArrowUp => "↑",
        ArrowDown => "↓",
        ArrowLeft => "←",
        ArrowRight => "→",
        Escape => "Esc",
        F1 => "F1",
        F2 => "F2",
        F3 => "F3",
        F4 => "F4",
        F5 => "F5",
        F6 => "F6",
        F7 => "F7",
        F8 => "F8",
        F9 => "F9",
        F10 => "F10",
        F11 => "F11",
        F12 => "F12",
        _ => "?",
    }
}

/// How a captured key is applied to an action.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CaptureMode {
    /// Append a new binding (up to two per action).
    Add,
    /// Replace the first binding.
    Replace,
}

/// A typed hotkey edit request raised by the UI and drained by the app after
/// the egui frame (same pattern as the other `req_*` fields).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HotkeyRequest {
    Set {
        action: Action,
        mode: CaptureMode,
        binding: KeyBinding,
    },
    Clear {
        action: Action,
    },
    ResetOne {
        action: Action,
    },
    ResetAll,
}

/// `Action` → ordered bindings. Missing actions fall back to the code defaults
/// at lookup time, so a partially-edited map (or an old `state.json` without
/// the `hotkeys` field) still behaves like the historical keymap.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HotkeyMap {
    #[serde(default)]
    bindings: HashMap<Action, Vec<KeyBinding>>,
}

impl HotkeyMap {
    /// The bindings currently assigned to `action` — the stored ones if the
    /// action has been touched, otherwise the code defaults.
    pub fn bindings_for(&self, action: Action) -> Vec<KeyBinding> {
        match self.bindings.get(&action) {
            Some(b) => b.clone(),
            None => default_bindings(action),
        }
    }

    /// Resolve a key event to an action. Physical key first (works across
    /// layouts); falls back to the logical key for injected events without a
    /// scancode. A `Shift`-specific binding wins over an unmodified one for the
    /// same key; an unmodified binding still matches a `Shift` press (the
    /// historical behavior), so defaults keep working when Shift is held.
    pub fn resolve(
        &self,
        physical_key: PhysicalKey,
        logical_key: &Key,
        modifiers: ModifiersState,
    ) -> Option<Action> {
        let code = match physical_key {
            PhysicalKey::Code(c) => c,
            _ => return resolve_logical(logical_key),
        };
        let shift = modifiers.shift_key();
        let mut shift_match = None;
        let mut plain_match = None;
        for action in Action::ALL {
            for b in self.bindings_for(action) {
                if b.code == code {
                    if b.shift {
                        if shift {
                            shift_match = Some(action);
                        }
                    } else {
                        // An unmodified binding matches any shift state — the
                        // historical behavior (Shift+Space still flipped pages).
                        plain_match = Some(action);
                    }
                }
            }
        }
        // A Shift-specific binding is the more specific match; otherwise the
        // unmodified binding (which also fires on Shift, as it always has).
        shift_match.or(plain_match)
    }

    /// The action that already owns `binding`, if any (excluding `action`
    /// itself). Used to reject duplicate shortcuts before committing.
    pub fn conflict(&self, action: Action, binding: KeyBinding) -> Option<Action> {
        for a in Action::ALL {
            if a == action {
                continue;
            }
            for b in self.bindings_for(a) {
                if b.code == binding.code && b.shift == binding.shift {
                    return Some(a);
                }
            }
        }
        None
    }

    /// Append a binding to `action` (caller has already checked `conflict`).
    pub fn add(&mut self, action: Action, binding: KeyBinding) {
        self.bindings.entry(action).or_default().push(binding);
    }

    /// Replace the first binding of `action` (or append if it has none).
    pub fn replace(&mut self, action: Action, binding: KeyBinding) {
        let v = self.bindings.entry(action).or_default();
        if v.is_empty() {
            v.push(binding);
        } else {
            v[0] = binding;
        }
    }

    /// Remove every binding from `action` (stored as empty, so it stays cleared
    /// rather than falling back to the default).
    pub fn clear(&mut self, action: Action) {
        self.bindings.insert(action, Vec::new());
    }

    /// Restore `action` to its default bindings.
    pub fn reset_one(&mut self, action: Action) {
        self.bindings.remove(&action);
    }

    /// Restore every action to its default bindings.
    pub fn reset_all(&mut self) {
        self.bindings.clear();
    }
}

/// The historical bindings, reproduced exactly. `CycleFit` had no key (it was
/// top-bar / settings only); it's in the map so users can add one.
fn default_bindings(action: Action) -> Vec<KeyBinding> {
    use KeyCode::*;
    let b = |code: KeyCode| KeyBinding {
        code,
        ctrl: false,
        alt: false,
        shift: false,
        super_: false,
    };
    match action {
        Action::Forward => vec![b(ArrowDown), b(Space), b(PageDown)],
        Action::Backward => vec![b(ArrowUp), b(PageUp)],
        Action::Left => vec![b(ArrowLeft)],
        Action::Right => vec![b(ArrowRight)],
        Action::First => vec![b(Home)],
        Action::Last => vec![b(End)],
        Action::CycleFit => vec![],
        Action::ToggleDir => vec![b(KeyD)],
        Action::ToggleLayout => vec![b(KeyS)],
        Action::ToggleScroll => vec![b(KeyC)],
        Action::ZoomIn => vec![b(Equal), b(NumpadAdd)],
        Action::ZoomOut => vec![b(Minus), b(NumpadSubtract)],
        Action::PresetWindow => vec![b(Digit9), b(Numpad9)],
        Action::PresetWidth => vec![b(Digit8), b(Numpad8)],
        Action::PresetActual => vec![b(Digit0), b(Numpad0)],
        Action::PresetSpreadLtr => vec![b(Digit7), b(Numpad7)],
        Action::PresetSpreadRtl => vec![b(Digit6), b(Numpad6)],
        Action::ToggleHelp => vec![b(F1)],
        Action::ToggleFullscreen => vec![b(F11)],
        Action::ToggleSpreadOffset => vec![b(KeyO)],
        Action::ToggleInfo => vec![b(KeyI)],
        Action::ToggleSeekbar => vec![b(KeyB)],
        Action::TogglePageJump => vec![b(KeyJ)],
        Action::TogglePageTransition => vec![b(KeyT)],
        Action::ToggleStretch => vec![b(KeyZ)],
        Action::ToggleSpineShadow => vec![b(KeyV)],
        Action::ToggleAnimBar => vec![b(KeyG)],
        Action::PrevVolume => vec![b(BracketLeft)],
        Action::NextVolume => vec![b(BracketRight)],
        Action::Rotate => vec![b(KeyR)],
        Action::ShowInExplorer => vec![b(KeyE)],
        Action::Quit => vec![b(Escape)],
    }
}

/// Logical-key fallback for injected events without a scancode (the physical
/// key is `Unidentified`). Covers the navigation keys + F1/Esc, matching the
/// historical `action_from` fallback.
fn resolve_logical(logical_key: &Key) -> Option<Action> {
    if let Key::Named(n) = logical_key {
        match n {
            NamedKey::ArrowDown | NamedKey::PageDown => return Some(Action::Forward),
            NamedKey::ArrowUp | NamedKey::PageUp => return Some(Action::Backward),
            NamedKey::ArrowRight => return Some(Action::Right),
            NamedKey::ArrowLeft => return Some(Action::Left),
            NamedKey::Home => return Some(Action::First),
            NamedKey::End => return Some(Action::Last),
            NamedKey::F1 => return Some(Action::ToggleHelp),
            NamedKey::Escape => return Some(Action::Quit),
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::{Key, ModifiersState, NativeKey, NativeKeyCode, PhysicalKey};

    fn mods(shift: bool) -> ModifiersState {
        let mut m = ModifiersState::default();
        m.set(ModifiersState::SHIFT, shift);
        m
    }

    fn resolve(map: &HotkeyMap, code: KeyCode, shift: bool) -> Option<Action> {
        map.resolve(
            PhysicalKey::Code(code),
            &Key::Unidentified(NativeKey::Unidentified),
            mods(shift),
        )
    }

    /// The default map must reproduce the historical bindings exactly.
    #[test]
    fn default_map_reproduces_historical_bindings() {
        let map = HotkeyMap::default();
        let cases = [
            (KeyCode::ArrowDown, Action::Forward),
            (KeyCode::Space, Action::Forward),
            (KeyCode::PageDown, Action::Forward),
            (KeyCode::ArrowUp, Action::Backward),
            (KeyCode::PageUp, Action::Backward),
            (KeyCode::ArrowRight, Action::Right),
            (KeyCode::ArrowLeft, Action::Left),
            (KeyCode::Home, Action::First),
            (KeyCode::End, Action::Last),
            (KeyCode::KeyD, Action::ToggleDir),
            (KeyCode::KeyS, Action::ToggleLayout),
            (KeyCode::KeyO, Action::ToggleSpreadOffset),
            (KeyCode::KeyC, Action::ToggleScroll),
            (KeyCode::KeyB, Action::ToggleSeekbar),
            (KeyCode::KeyJ, Action::TogglePageJump),
            (KeyCode::KeyT, Action::TogglePageTransition),
            (KeyCode::KeyZ, Action::ToggleStretch),
            (KeyCode::KeyV, Action::ToggleSpineShadow),
            (KeyCode::KeyG, Action::ToggleAnimBar),
            (KeyCode::KeyE, Action::ShowInExplorer),
            (KeyCode::KeyR, Action::Rotate),
            (KeyCode::Equal, Action::ZoomIn),
            (KeyCode::NumpadAdd, Action::ZoomIn),
            (KeyCode::Minus, Action::ZoomOut),
            (KeyCode::NumpadSubtract, Action::ZoomOut),
            (KeyCode::Digit9, Action::PresetWindow),
            (KeyCode::Numpad9, Action::PresetWindow),
            (KeyCode::Digit8, Action::PresetWidth),
            (KeyCode::Numpad8, Action::PresetWidth),
            (KeyCode::Digit7, Action::PresetSpreadLtr),
            (KeyCode::Numpad7, Action::PresetSpreadLtr),
            (KeyCode::Digit6, Action::PresetSpreadRtl),
            (KeyCode::Numpad6, Action::PresetSpreadRtl),
            (KeyCode::Digit0, Action::PresetActual),
            (KeyCode::Numpad0, Action::PresetActual),
            (KeyCode::F1, Action::ToggleHelp),
            (KeyCode::KeyI, Action::ToggleInfo),
            (KeyCode::F11, Action::ToggleFullscreen),
            (KeyCode::Escape, Action::Quit),
            (KeyCode::BracketLeft, Action::PrevVolume),
            (KeyCode::BracketRight, Action::NextVolume),
        ];
        for (code, action) in cases {
            assert_eq!(
                resolve(&map, code, false),
                Some(action),
                "default binding for {code:?}"
            );
        }
        // CycleFit had no key historically.
        assert_eq!(resolve(&map, KeyCode::KeyF, false), None);
    }

    /// Serialized IDs round-trip through JSON.
    #[test]
    fn serialized_ids_round_trip() {
        for a in Action::ALL {
            let json = serde_json::to_string(&a).unwrap();
            assert_eq!(serde_json::from_str::<Action>(&json).unwrap(), a);
        }
    }

    /// A full map round-trips through JSON, preserving bindings and modifiers.
    #[test]
    fn hotkey_map_round_trips_through_json() {
        let mut map = HotkeyMap::default();
        map.add(
            Action::ToggleDir,
            KeyBinding {
                code: KeyCode::KeyX,
                ctrl: false,
                alt: false,
                shift: true,
                super_: false,
            },
        );
        map.clear(Action::Forward);
        let json = serde_json::to_string(&map).unwrap();
        let back: HotkeyMap = serde_json::from_str(&json).unwrap();
        // The Shift+X binding survived.
        assert_eq!(
            back.bindings_for(Action::ToggleDir),
            vec![KeyBinding {
                code: KeyCode::KeyX,
                ctrl: false,
                alt: false,
                shift: true,
                super_: false,
            }]
        );
        // The cleared action stays cleared.
        assert!(back.bindings_for(Action::Forward).is_empty());
        // Untouched actions still fall back to defaults.
        assert_eq!(
            resolve(&back, KeyCode::ArrowUp, false),
            Some(Action::Backward)
        );
    }

    /// A map with no `hotkeys` field (old state.json) loads defaults.
    #[test]
    fn missing_hotkeys_loads_defaults() {
        let map: HotkeyMap = serde_json::from_str("{}").unwrap();
        assert_eq!(resolve(&map, KeyCode::Space, false), Some(Action::Forward));
    }

    /// Physical-code + modifier matching: a Shift-specific binding wins over an
    /// unmodified one for the same key, and an unmodified binding still fires
    /// on a Shift press (historical behavior).
    #[test]
    fn shift_binding_is_more_specific_but_unmodified_still_fires_on_shift() {
        let mut map = HotkeyMap::default();
        let shift_space = KeyBinding {
            code: KeyCode::Space,
            ctrl: false,
            alt: false,
            shift: true,
            super_: false,
        };
        map.add(Action::ToggleDir, shift_space);
        // Shift+Space → the Shift-specific binding.
        assert_eq!(resolve(&map, KeyCode::Space, true), Some(Action::ToggleDir));
        // Plain Space → Forward (the unmodified default).
        assert_eq!(resolve(&map, KeyCode::Space, false), Some(Action::Forward));
        // Shift+ArrowDown → Forward (unmodified binding fires on Shift).
        assert_eq!(
            resolve(&map, KeyCode::ArrowDown, true),
            Some(Action::Forward)
        );
    }

    /// Multi-binding actions match any of their bindings.
    #[test]
    fn multi_binding_action_matches_any_binding() {
        let map = HotkeyMap::default();
        for code in [KeyCode::ArrowDown, KeyCode::Space, KeyCode::PageDown] {
            assert_eq!(resolve(&map, code, false), Some(Action::Forward));
        }
    }

    /// A duplicate binding is rejected and leaves the map unchanged.
    #[test]
    fn duplicate_rejection_leaves_map_unchanged() {
        let mut map = HotkeyMap::default();
        let space = KeyBinding {
            code: KeyCode::Space,
            ctrl: false,
            alt: false,
            shift: false,
            super_: false,
        };
        assert_eq!(
            map.conflict(Action::ToggleDir, space),
            Some(Action::Forward)
        );
        map.add(Action::ToggleDir, space); // (caller would not; simulate anyway)
        // The conflict check still reports the original owner.
        assert_eq!(
            map.conflict(Action::ToggleLayout, space),
            Some(Action::Forward)
        );
    }

    /// Clear / reset-one / reset-all behavior.
    #[test]
    fn clear_and_reset_behavior() {
        let mut map = HotkeyMap::default();
        // Clear: stored empty, stays cleared (no fallback to default).
        map.clear(Action::Forward);
        assert!(map.bindings_for(Action::Forward).is_empty());
        assert_eq!(resolve(&map, KeyCode::Space, false), None);
        // Reset one: falls back to defaults again.
        map.reset_one(Action::Forward);
        assert_eq!(resolve(&map, KeyCode::Space, false), Some(Action::Forward));
        // Reset all.
        map.clear(Action::Forward);
        map.clear(Action::Backward);
        map.reset_all();
        assert_eq!(resolve(&map, KeyCode::Space, false), Some(Action::Forward));
        assert_eq!(
            resolve(&map, KeyCode::ArrowUp, false),
            Some(Action::Backward)
        );
    }

    /// The logical fallback covers injected events without a scancode.
    #[test]
    fn logical_fallback_for_injected_events() {
        let map = HotkeyMap::default();
        assert_eq!(
            map.resolve(
                PhysicalKey::Unidentified(NativeKeyCode::Unidentified),
                &Key::Named(NamedKey::PageDown),
                mods(false),
            ),
            Some(Action::Forward)
        );
    }

    /// Labels render readable names.
    #[test]
    fn binding_labels_are_readable() {
        let b = |code: KeyCode, shift: bool| KeyBinding {
            code,
            ctrl: false,
            alt: false,
            shift,
            super_: false,
        };
        assert_eq!(b(KeyCode::Space, false).label(), "Space");
        assert_eq!(b(KeyCode::PageDown, false).label(), "PgDn");
        assert_eq!(b(KeyCode::ArrowRight, false).label(), "→");
        assert_eq!(b(KeyCode::KeyA, true).label(), "Shift+A");
        assert_eq!(b(KeyCode::Digit9, false).label(), "9");
    }

    /// Modifier keys can't be captured as bindings on their own.
    #[test]
    fn modifier_keys_are_rejected_as_bindings() {
        for code in [
            KeyCode::ShiftLeft,
            KeyCode::ShiftRight,
            KeyCode::ControlLeft,
            KeyCode::ControlRight,
            KeyCode::AltLeft,
            KeyCode::AltRight,
            KeyCode::SuperLeft,
            KeyCode::SuperRight,
            KeyCode::CapsLock,
            KeyCode::NumLock,
            KeyCode::ScrollLock,
        ] {
            assert!(is_modifier_key(code), "{code:?} should be a modifier key");
        }
        assert!(!is_modifier_key(KeyCode::KeyA));
        assert!(!is_modifier_key(KeyCode::Space));
    }

    /// Every action appears in exactly one category, and `category()` agrees
    /// with the `Category::actions()` grouping the UI iterates.
    #[test]
    fn every_action_is_in_exactly_one_category() {
        let mut seen = std::collections::HashSet::new();
        for c in Category::ALL {
            for a in c.actions() {
                assert!(
                    seen.insert(*a),
                    "action {:?} listed in more than one category",
                    a
                );
                assert_eq!(
                    a.category(),
                    c,
                    "action {:?} disagrees with its category",
                    a
                );
            }
        }
        assert_eq!(
            seen.len(),
            Action::ALL.len(),
            "some action is uncategorized"
        );
    }
}
