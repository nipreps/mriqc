#!/bin/bash
#
# Balance nipype testing workflows across CircleCI build nodes
#

# Setting      # $ help set
set -e         # Exit immediately if a command exits with a non-zero status.
set -u         # Treat unset variables as an error when substituting.
set -x         # Print command traces before executing command.

git log --grep="Merge pull request" `git describe --tags --abbrev=0`..HEAD --pretty='format:  * %b %s' | sed  's/Merge pull request \#\([^\d]*\)\ from\ .*/(\#\1)/' >> newchanges
echo "" >> newchanges
echo "" >> newchanges
cat CHANGES.txt >> newchanges 
mv newchanges CHANGES.txt