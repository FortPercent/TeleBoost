.PHONY: test serve report clean\

install:
	pip install -r requirements.txt
	apt install -y openjdk-11-jdk
	dpkg -i allure_2.34.0-1_all.deb

test:
	pytest -ra -q --alluredir=allure-results

serve:
	allure serve ./allure-results

report:
	allure generate allure-results -o allure-report --clean

clean:
	rm -rf allure-results allure-report
