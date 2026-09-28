test:
	PYTHONPATH=backend python -m unittest discover -s tests -v
lint:
	python -m compileall -q backend tools
up:
	docker compose up -d --build
smoke:
	sh tools/smoke_test.sh
