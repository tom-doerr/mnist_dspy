#!/bin/bash

# Run pytest with verbose output and generate XML report
#pytest -v --runslow --junitxml=test_results.xml 2>&1 | tee test_output.txt
pytest --junitxml=test_results.xml

# Print summary message
echo -e "\nTest results saved to:"
echo "- XML report: test_results.xml"
echo "- Console output: test_output.txt"
