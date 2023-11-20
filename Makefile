hub_repo := cacophonyproject/audio-analysis

.PHONY: docker
docker:
	docker build -t $(hub_repo) .

.PHONY: docker-no-cache
docker-no-cache:
	docker build --no-cache -t $(hub_repo) .

.PHONY: docker-push
docker-push: docker
	docker tag $(hub_repo) $(hub_repo):$(TAG)
	docker push $(hub_repo):$(TAG)
	docker push $(hub_repo):latest
