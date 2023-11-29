if [ -z $1 ];then
    cat $(ls -1 ret/ret-*.err | sort -n -t - -k 2 | tail -n 1)
else
    cat ret/ret-"$1".err
fi