#!/bin/sh

sudo chown -R _mysql:mysql /usr/local/var/mysql

if [[ $1 = 'start' ]]
then
    sudo mysql.server start
elif [[ $1 = 'stop' ]]
then
    sudo mysql.server stop
else
    echo ERROR: bad instruction
fi

# enhance this to include command line args

