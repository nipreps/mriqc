#!/bin/bash
#
# Collects the pull-requests since the latest release and
# aranges them in the CHANGES.txt file.
#
# This is a script to be run before releasing a new version.
#
#

# Setting      # $ help set
set -e         # Exit immediately if a command exits with a non-zero status.
set -u         # Treat unset variables as an error when substituting.
set -x         # Print command traces before executing command.

echo "$1" >> newchanges
echo $( printf "%${#1}s" | tr " " "=" ) >> newchanges
echo "" >> newchanges

git log --grep="Merge pull request" `git describe --tags --abbrev=0`..HEAD --pretty='format:  * %b %s' | sed  's/Merge pull request \#\([^\d]*\)\ from\ .*/(\#\1)/' >> newchanges
echo "" >> newchanges
echo "" >> newchanges
cat CHANGES.txt >> newchanges 
mv newchanges CHANGES.txt