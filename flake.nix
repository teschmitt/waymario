{
  description = "waymario — autonomous Mario Kart 64 Rainbow Road driver (HDMI capture -> CV -> N64 controller)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs:
        let
          python = pkgs.python313;

          # Native libraries the opencv-python wheel dlopen()s at runtime.
          # uv installs the prebuilt wheel; nix has to supply these on the loader path.
          runtimeLibs = with pkgs; [
            stdenv.cc.cc.lib # libstdc++
            zlib
            glib
          ] ++ lib.optionals stdenv.isLinux [
            libGL
            libGLU
            xorg.libX11
            xorg.libXext
            xorg.libXrender
            xorg.libSM
            xorg.libICE
          ];
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.uv
              python
            ];

            env = {
              # Use the nix-provided interpreter; don't let uv download its own.
              UV_PYTHON = python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              echo "waymario dev shell — run 'uv sync' then 'uv run waymario --help'"
            '';
          };
        });
    };
}
