#!/usr/bin/env bash
set -euo pipefail

# Navigate to project root
cd "$(dirname "$0")/.."

CMD=${1:-up}
MODE=${MODE:-standalone}

# Select compose file based on MODE
if [ "$MODE" = "prod" ]; then
  compose="docker-compose.prod.yml"
elif [ "$MODE" = "standalone" ]; then
  compose="docker-compose.standalone.yml"
else
  compose="docker-compose.yml"
fi

case "$CMD" in
  up)    docker compose -f "$compose" up --build ;;
  down)  docker compose -f "$compose" down ;;
  build) docker compose -f "$compose" build --no-cache ;;
  logs)  docker compose -f "$compose" logs -f ;;
  *) 
    echo "Usage: MODE=[standalone|prod|default] ./dev.sh [up|down|build|logs]"
    echo "Examples:"
    echo "  ./dev.sh up                    # Start with docker-compose.yml"
    echo "  MODE=standalone ./dev.sh up    # Start standalone mode (no auth)"
    echo "  MODE=prod ./dev.sh up          # Start production mode (with auth)"
    exit 1 
    ;; 
esac
