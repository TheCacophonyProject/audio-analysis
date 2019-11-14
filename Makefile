.PHONY: docker
docker:
	docker build -t audio-analysis .

.PHONY: docker-no-cache
docker-no-cache:
	docker build --no-cache -t audio-analysis .

.PHONY: docker-release
docker-release: docker
	docker tag audio-analysis cacophonyproject/audio-analysis:latest
	docker push cacophonyproject/audio-analysis:latest
