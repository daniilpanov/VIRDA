from virda.main import main

if __name__ == "__main__":
    result = main()
    print(f"Stage 1: mesh with {len(result.mesh.vertices)} vertices")
