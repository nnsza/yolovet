#[cfg(target_os = "windows")]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=resources/app_icon.ico");
    let mut res = winres::WindowsResource::new();
    res.set_icon("resources/app_icon.ico");
    if let Err(err) = res.compile() {
        panic!("failed to compile Windows resources: {err}");
    }
}

#[cfg(not(target_os = "windows"))]
fn main() {}
