FROM golang:1.8.2

RUN mkdir -p /opt/mlkube
COPY mx_operator /opt/mlkube
RUN chmod a+x /opt/mlkube/mx_operator

ENTRYPOINT ["/opt/mlkube/mx_operator", "-alsologtostderr"]
