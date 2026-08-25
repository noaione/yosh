//! yosh-thumbnail — a Windows Explorer thumbnail provider for comic archives.
//!
//! This is an in-process COM server DLL (`cdylib`) that Explorer's thumbnail
//! cache loads to paint previews for `.cbz` / `.cbr` / `.cb7` files. It exposes
//! [`IThumbnailProvider`] (the controller) and [`IInitializeWithStream`] (Shell
//! process isolation — the provider never sees a filesystem path, only a
//! `IStream` of the file's bytes), plus the `DllGetClassObject` /
//! `DllCanUnloadNow` entry points Windows expects.
//!
//! Everything is Windows-only. On other targets the crate compiles to an empty
//! library so a cross-platform `cargo check`/`build` of the workspace works.

#[cfg(windows)]
mod thumbnail;
