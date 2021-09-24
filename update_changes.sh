#!/bin/bash
#
# Collects the pull-requests since the latest release and
# aranges them in the CHANGES.rst file.
#
# This is a script to be run before releasing a new version.
#
# Usage /bin/bash update_changes.sh 1.0.1
#

# Setting      # $ help set
set -u         # Treat unset variables as an error when substituting.
set -x         # Print command traces before executing command.

# Check whether the Upcoming release header is present
head -1 CHANGES.rst | grep -q Upcoming
UPCOMING=$?
if [[ "$UPCOMING" == "0" ]]; then
    head -n3  CHANGES.rst >> newchanges
fi

# Search for PRs since previous release
git show --pretty='format:  * %b %s'  HEAD | sed 's/Merge pull request \#\([^\d]*\)\ from\ .*/(\#\1)/' >> newchanges

# Add back the Upcoming header if it was present
if [[ "$UPCOMING" == "0" ]]; then
    tail -n+4 CHANGES.rst >> newchanges
else
    cat CHANGES.rst >> newchanges
fi

# Replace old CHANGES.rst with new file
mv newchanges CHANGES.rst

