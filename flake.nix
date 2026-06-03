{
  description = "Bindings between Numpy and Eigen using Boost.Python";

  inputs = {
    gepetto.url = "github:gepetto/nix";
    jrl-cmakemodules = {
      url = "github:jrl-umi3218/jrl-cmakemodules";
      inputs.gepetto.follows = "gepetto";
    };
  };

  outputs =
    inputs:
    inputs.gepetto.lib.mkFlakoboros inputs (
      { lib, ... }:
      {
        overlays = [ inputs.jrl-cmakemodules.overlays.flakoboros ];
        pyOverrideAttrs.eigenpy = {
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./CMakeLists.txt
              ./doc
              ./include
              ./package.xml
              ./python
              ./src
              ./unittest
            ];
          };
        };
        extends.eigen5 = final: prev: {
          eigen = final.eigen_5;
          pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
            (_python-final: python-prev: {
              scipy = python-prev.scipy.overrideAttrs {
                # broken on linux arm
                doInstallCheck = false;
              };
            })
          ];
        };
      }
    );
}
