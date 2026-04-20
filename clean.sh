#!/bin/bash

REMOVE_VENV=false

for arg in "$@"; do
	case "$arg" in
		--remove-venv)
			REMOVE_VENV=true
			;;
	esac
done

if [ -z "${RL_PCB:-}" ]; then
	source setup.sh
fi

if [ "$REMOVE_VENV" = true ]; then
	DIR="${RL_PCB}/venv"
	echo -n "Attempting to clean ${DIR} ... "
	if [ -d "${DIR}" ]; then
		echo "Found, deleting."
		rm -fr "${DIR}"
	else
		echo "Not found, therefore nothing to clean."
	fi
else
	echo "Skipping ${RL_PCB}/venv (preserved by default)."
	echo "Use ./clean.sh --remove-venv if you want to delete it explicitly."
fi

DIR="${RL_PCB}/bin"
echo -n "Attempting to clean ${DIR} ... "
if [ -d "${DIR}" ]; then
	echo "Found, deleting."
	rm -fr "${DIR}"
else
	echo "Not found, therefore nothing to clean."
fi
