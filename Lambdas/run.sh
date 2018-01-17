#!/bin/bash

venv() {
  pip install virtualenv
  virtualenv ../env
}

activate() {
  source ../env/bin/activate
}


package() {
  activate
  npm run serverless package
  deactivate
}

deploy() {
  activate
  npm run serverless deploy
  deactivate
}


serverless_reqs() {
  cat serverless_plugins.txt | while read line
  do
    npm run sls plugin install -- -n $line
  done
}

node_reqs() {
  npm install
}


python_reqs() {
  ../env/bin/pip install -r requirements.txt
}


setup() {
  python_reqs
  node_reqs
  serverless_reqs
}

remove_stack() {
  npm run serverless remove
}

case $1 in
  deploy)
    deploy
    ;;
  package)
    package
    ;;
  setup)
    setup
    ;;
  remove_stack)
    remove_stack
    ;;
  activate)
    activate
    ;;
  venv)
    venv
    ;;
  *)
    echo $"Usage: $0 {package|deploy|setup|remove_stack}"
esac
