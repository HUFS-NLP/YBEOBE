0. Issue:
If you use Free account, you can use Azure translator API up to 2M characters per month.
Total characters in GoEmotions Train Dataset's text is 2,969,282.

1. Sign in to the Azure Portal:
If you don't have an Azure account, you will need to create one. You can sign up here. Azure often offers free credits to get started, but always be mindful of any costs that can accrue after those credits are used up.

Once you have an account, sign in to the Azure Portal.

2. Create a Translator Service (part of Azure Cognitive Services):
Once logged in, in the Azure Portal, click on Create a resource. This is usually a green '+' sign located in the top left corner of the dashboard.

In the search box, type "Translator" and select "Translator" from the dropdown list.

Click the Create button to start the process of setting up the Translator service.

Fill in the required information:

Subscription: Choose the Azure subscription you want to use.
Resource Group: You can create a new one or use an existing. Resource groups are a way to organize related Azure resources.
Region: Choose a region that is geographically closer to you or your users.
Name: Give a unique name to your Translator service.
Pricing Tier: Choose a pricing tier based on your needs. For testing purposes, the free or basic tier might suffice, but always check the details of what each tier offers.
Click Review + create, review your selections, and then click Create.

3. Obtain the subscription key and endpoint for the service:
After the Translator service is deployed, go to the Azure Portal dashboard.

In the left sidebar, click on Resource groups, then select the resource group where you created the Translator service.

Click on the name of the Translator service you just created.

In the left sidebar of the Translator service page, under the RESOURCE MANAGEMENT section, click on Keys and Endpoint.

Here, you'll find two keys (either can be used) and the endpoint for your service. You'll use these in your code to authenticate and make requests to the Translator API.

4. Create '.env' file in the project folder:

below is the format.
AZURE_TRANSLATOR_KEY = "xxxxxxxxx"

If your translator setting is different from below, change the code's header.
  'Ocp-Apim-Subscription-Region': 'eastasia',
  'Content-type': 'application/json',
  'X-ClientTraceId': str(uuid.uuid4())
