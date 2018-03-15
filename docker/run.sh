docker run -it --name  fabo_thesis \
    -p 11345:11345 \
    -p 11311:11311 \
    -p 8888:8888 \
    -v /home/fabo/fabo_thesis:/app \
    fabo-thesis