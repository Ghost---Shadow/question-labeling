from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

credentials = GoogleCredentials.get_application_default()

service = discovery.build("compute", "v1", credentials=credentials)

project = "angular-unison-350808"
zone = "us-central1-a"
instance = "q-labeling-2"

request = service.instances().stop(project=project, zone=zone, instance=instance)
response = request.execute()
print(response)
