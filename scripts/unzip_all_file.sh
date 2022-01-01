for i in *.zip; do
  newdir="${i:0:-4}" && mkdir "$newdir"
  unzip "$i" -d  "$newdir"
done