POSTGRESPATH = $(PWD)/../../../mnt/hda/postgres
MINIOPATH = $(PWD)/../../../mnt/hda/minio
PGADMINPATH = $(PWD)/../../../mnt/hda/pgadmin

build.train:
	docker compose -f docker-compose-train.yaml up --build -d

start.train:
	docker compose -f docker-compose-train.yaml up -d

down.train:
	docker compose -f docker-compose-train.yaml down -v

build.tracker:
	if [ ! -e "$(POSTGRESPATH)" ]; then \
		mkdir -p $(POSTGRESPATH); \
	fi
	
	if [ ! -e "$(MINIOPATH)" ]; then \
		mkdir -p $(MINIOPATH); \
	fi
	
	if [ ! -e "$(PGADMINPATH)" ]; then \
		mkdir -p $(PGADMINPATH); \
	fi
	

	docker compose -f docker-compose-tracker.yaml up --build -d

start.tracker:
	docker compose -f docker-compose-tracker.yaml up -d

down.tracker:
	docker compose -f docker-compose-tracker.yaml down -v --remove-orphans