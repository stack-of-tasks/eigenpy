:: Setup ccache
set CMAKE_CXX_COMPILER_LAUNCHER=ccache

:: Create compile_commands.json for language server
set CMAKE_EXPORT_COMPILE_COMMANDS=1

:: Activate color output with Ninja
set CMAKE_COLOR_DIAGNOSTICS=1
