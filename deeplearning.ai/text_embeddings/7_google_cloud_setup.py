# Google Cloud setup

This lab outlines the steps you will need to take to use Google Cloud and Vertex AI for your own projects.

**Note**: To try this out yourself, please download this notebook and run it on your local computer.

### Create a Google Cloud Project

Google Cloud projects form the basis for creating, enabling, and using all Google Cloud services including managing APIs, enabling billing, adding and removing collaborators, and managing permissions for Google Cloud resources.

Your usage of Google Cloud tools is always associated with a project.


You will be prompted to create a new project the first time you visit the [Cloud Console](https://console.cloud.google.com)

   > Note that you can create a [free project](https://cloud.google.com/free/docs/gcp-free-tier) which includes a 90-day $300 Free Trial.

Learn more about projects [here.](https://cloud.google.com/resource-manager/docs/creating-managing-projects)

### Set up Billing

A Cloud Billing account is used to define who pays for a given set of resources, and it can be linked to one or more projects. Project usage is charged to the linked Cloud Billing account.

Within your project, you can configure billing by selecting "Billing" in the menu on the left.

![select billing](images/billing.png)



Make sure that **billing is enabled** for your Google Cloud project, [_click here_](https://cloud.google.com/billing/docs/how-to/modify-project) to learn how to confirm that billing is enabled.

### Enable APIs

Once you have a project set up with a billing account, you will need to enable any services you want to use.

[_Click here_](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform,iam,bigquery.googleapis.com)
   to **enable the following APIs** in your Google Cloud project:

- _Vertex AI_
- _BigQuery_
- _IAM_

### Create service account

A service account is a special kind of account typically used by an application or compute workload, such as a Compute Engine instance, rather than a person. A service account is identified by its email address, which is unique to the account. To learn more, check out [this intro video](https://www.youtube.com/watch?v=xXk1YlkKW_k).

You will need to create a service account and give it access to the Google Cloud services you want to use.

#### 1. Go to the [Create Service Account](https://console.cloud.google.com/projectselector/iam-admin/serviceaccounts/create?walkthrough_id=iam--create-service-account&_ga=2.184446630.1022340625.1692280734-338572696.1692280734#step_index=1) page and **select your project**

#### 2. Give the account a name (you can pick anything)

![create service account](images/create_sa.png)

#### Grant the account the following permissions

![grant permissions](images/sa_permissions.png)

### Create Service Account key

Once you have created your service account, you need to create a key.

#### 1. Select your newly created service account then click. ADD KEY -> create new key.

#### 2. Select JSON key type and click create

![JSON key](images/json_key.png)


Clicking Create downloads a service account key file. After you download the key file, you cannot download it again.

### Create credentials

To use Vertex AI services, you will need to authenticate with your credentials.

Using the json file you just downloaded, you will create a credentials object.

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

# Path to your service account key file
key_path = 'your_key.json' #Path to the json key associated with your service account from google cloud

# Create credentials object

credentials = Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform'])

if credentials.expired:
    credentials.refresh(Request())

### Connect to Vertex

Once you have a project and your credentials, you can use Vertex AI tools.

[**Copy your project ID**](https://cloud.google.com/resource-manager/docs/creating-managing-projects) and paste it in the `PROJECT_ID` field below.

PROJECT_ID = 'your_project_ID'
REGION = 'us-central1'

import vertexai

# initialize vertex
vertexai.init(project = PROJECT_ID, location = REGION, credentials = credentials)
