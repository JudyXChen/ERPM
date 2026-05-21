# Usage:
#   scripts/run_in_smart.sh python3 -m model.run --glu glut_release --voltage -70
set -euo pipefail

IMAGE="ghcr.io/rangamanilabucsd/smart:latest"
SMART_MA="/usr/lib/python3/dist-packages/smart/model_assembly.py"

# --ulimit stack=-1: the 100-state RyR makes the UFL form so deeply nested
# that FEniCS form compilation overflows the default 8 MB C stack and
# SIGSEGVs (even with a raised Python recursionlimit). Allow an unlimited
# container stack; raise it inside too.
# shellcheck disable=SC2068
docker run --rm --ulimit stack=-1:-1 -v "$(pwd)":/repo -w /repo "$IMAGE" bash -lc '
set -e
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576 2>/dev/null || true
sed -i "s/isinstance(v, (numbers.Number, None))/isinstance(v, (numbers.Number, type(None)))/" '"$SMART_MA"'
sed -i "s|        initial_equation_units = self.equation_lambda_eval(\"units\")|        self._expected_flux_units = 1.0 * unit.dimensionless  # PATCH: dimensionless Step-1 model\n        initial_equation_units = self.equation_lambda_eval(\"units\")|" '"$SMART_MA"'
exec "$@"
' bash "$@"