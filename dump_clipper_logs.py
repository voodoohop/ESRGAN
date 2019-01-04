from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.pytorch import deploy_pytorch_model
from torch import nn

clipper_conn = ClipperConnection(DockerContainerManager())
clipper_conn.connect()

log_files = clipper_conn.get_clipper_logs()

logs = [open(filename).read() for filename in log_files]

# for log in logs:
#     print(log)

print(log_files )