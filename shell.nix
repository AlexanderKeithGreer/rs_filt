with import <nixpkgs> {};
let
  unstable = import <nixos-unstable> { config = { allowUnfree = true; }; };
in {
     testEnv = unstable.stdenv.mkDerivation {
       name = "helloTest";
       buildInputs = [unstable.stdenv
                      unstable.cargo unstable.rustc unstable.rustup
		      #project specific deps
		      unstable.pkg-config unstable.freetype unstable.fontconfig

                      ];
     };
}
