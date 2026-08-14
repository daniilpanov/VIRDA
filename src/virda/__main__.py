from virda.main import run

if __name__ == "__main__":
    result = run()
    print(f"Stage 1: mesh with {len(result.mesh.vertices)} vertices")
