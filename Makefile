TICK ?= 0
COMPOSE = docker-compose.yml
OVERRIDE = $(if $(filter $(TICK),1),docker-compose.tick.yml,docker-compose.no-tick.yml)
.PHONY: up build down
up:    ; docker compose -f $(COMPOSE) -f $(OVERRIDE) up --build
build: ; docker compose -f $(COMPOSE) -f $(OVERRIDE) build --no-cache
down:  ; docker compose -f $(COMPOSE) -f $(OVERRIDE) down -v
