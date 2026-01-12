#!/usr/bin/env python3
"""
Diagnostic script to verify FFmpeg integration with av and torchvision.
Run this to ensure both libraries pick up the Nix-installed FFmpeg 7.x.
"""

import shutil
import subprocess


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}\n")


def main() -> None:
    # ─────────────────────────────────────────────────────────────
    # System FFmpeg
    # ─────────────────────────────────────────────────────────────
    section("System FFmpeg")

    ffmpeg_path = shutil.which("ffmpeg")
    print(f"ffmpeg binary: {ffmpeg_path}")

    if ffmpeg_path:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        version_line = result.stdout.split("\n")[0]
        print(f"ffmpeg version: {version_line}")

        # Check for libsvtav1 encoder (required by lerobot)
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
        )
        has_svtav1 = "libsvtav1" in result.stdout
        print(f"libsvtav1 encoder available: {'✓ YES' if has_svtav1 else '✗ NO'}")
    else:
        print("⚠️  ffmpeg not found in PATH!")

    # ─────────────────────────────────────────────────────────────
    # PyAV (av)
    # ─────────────────────────────────────────────────────────────
    section("PyAV (av module)")

    try:
        import av

        print(f"av version: {av.__version__}")
        print(f"av library versions:")
        print(
            f"  libavcodec:  {av.codec.Codec.__module__}"
        )  # just to confirm import works

        # Get the actual FFmpeg library versions PyAV was built against
        try:
            # av.library_versions gives us the actual linked library versions
            lib_versions = av.library_versions
            for lib, version in lib_versions.items():
                if isinstance(version, tuple):
                    # Newer av versions return tuples (major, minor, micro)
                    major, minor, micro = version
                else:
                    # Older av versions return packed integers
                    major = (version >> 16) & 0xFF
                    minor = (version >> 8) & 0xFF
                    micro = version & 0xFF
                print(f"  {lib}: {major}.{minor}.{micro}")
        except AttributeError:
            print("  (library_versions not available)")

        # Check if libsvtav1 encoder is available via av
        try:
            enc = av.codec.Codec("libsvtav1", "w")
            print(f"\nlibsvtav1 encoder via av: ✓ available ({enc.long_name})")
        except av.codec.codec.UnknownCodecError:
            print("\nlibsvtav1 encoder via av: ✗ NOT available")
        except Exception as e:
            print(f"\nlibsvtav1 encoder check error: {e}")

        # Try a simple encode/decode test
        print("\nQuick encode test:")
        try:
            container = av.open("pipe:", mode="w", format="mp4")
            stream = container.add_stream("h264", rate=30)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            container.close()
            print("  h264 encoding: ✓ works")
        except Exception as e:
            print(f"  h264 encoding: ✗ failed - {e}")

    except ImportError as e:
        print(f"✗ Failed to import av: {e}")
        print("\nTo install av with system FFmpeg:")
        print("  pip install av --no-binary av")

    # ─────────────────────────────────────────────────────────────
    # torchvision
    # ─────────────────────────────────────────────────────────────
    section("torchvision")

    try:
        import torchvision

        print(f"torchvision version: {torchvision.__version__}")

        # Check video backend
        try:
            from torchvision.io import VideoReader  # noqa: F401

            print(f"VideoReader available: ✓ YES")
        except ImportError as e:
            print(f"VideoReader available: ✗ NO ({e})")

        # Check available video backends
        try:
            backends = torchvision.io.video._video_opt.get_video_backend()
            print(f"Video backend: {backends}")
        except AttributeError:
            pass

        # Try to get video capabilities
        try:
            from torchvision.io import _video_opt  # noqa: F401

            print(f"_video_opt module: ✓ available")
        except ImportError:
            print(f"_video_opt module: ✗ not available")

        # Check if pyav backend works
        try:
            print("\nTesting pyav backend:")
            torchvision.set_video_backend("pyav")
            print(f"  pyav backend: ✓ set successfully")
        except Exception as e:
            print(f"  pyav backend: ✗ failed - {e}")

    except ImportError as e:
        print(f"✗ Failed to import torchvision: {e}")

    # ─────────────────────────────────────────────────────────────
    # Environment variables
    # ─────────────────────────────────────────────────────────────
    section("Relevant Environment Variables")

    import os

    env_vars = [
        "PKG_CONFIG_PATH",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "FFMPEG_LIBRARY_PATH",
        "PATH",
    ]

    for var in env_vars:
        value = os.environ.get(var)
        if value:
            # Truncate long paths for readability
            paths = value.split(":")
            if len(paths) > 3:
                display = f"{paths[0]}:... ({len(paths)} entries)"
            else:
                display = value
            print(f"{var}:\n  {display}")
        else:
            print(f"{var}: (not set)")

    # ─────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────
    section("Summary")

    issues = []

    if not ffmpeg_path or "/nix/" not in ffmpeg_path:
        issues.append("FFmpeg binary not from Nix store")

    try:
        import av

        lib_versions = av.library_versions
        avcodec_version = lib_versions.get("libavcodec", 0)
        if isinstance(avcodec_version, tuple):
            major = avcodec_version[0]
        else:
            major = (avcodec_version >> 16) & 0xFF
        if major < 60:  # FFmpeg 7.x has libavcodec 61.x
            issues.append(
                f"PyAV linked against old FFmpeg (libavcodec {major}.x, need 60+)"
            )
    except Exception as _:
        issues.append("Could not verify PyAV FFmpeg version")

    if issues:
        print("⚠️  Potential issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nTip: If av is using wrong FFmpeg, reinstall with:")
        print("  uv pip install av --no-binary av --force-reinstall")
    else:
        print("✓ Everything looks good!")


if __name__ == "__main__":
    main()
