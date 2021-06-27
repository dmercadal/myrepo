<!-----
NEW: Check the "Suppress top comment" option to remove this info from the output.

Conversion time: 3.146 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β29
* Sun Jun 27 2021 09:11:30 GMT-0700 (PDT)
* Source doc: DP-100 Exam Prep
* This document has images: check for >>>>>  gd2md-html alert:  inline image link in generated source and store images to your server. NOTE: Images in exported zip file from Google Docs may not appear in  the same order as they do in your doc. Please check the images!

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 0; ALERTS: 23.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>
<a href="#gdcalert2">alert2</a>
<a href="#gdcalert3">alert3</a>
<a href="#gdcalert4">alert4</a>
<a href="#gdcalert5">alert5</a>
<a href="#gdcalert6">alert6</a>
<a href="#gdcalert7">alert7</a>
<a href="#gdcalert8">alert8</a>
<a href="#gdcalert9">alert9</a>
<a href="#gdcalert10">alert10</a>
<a href="#gdcalert11">alert11</a>
<a href="#gdcalert12">alert12</a>
<a href="#gdcalert13">alert13</a>
<a href="#gdcalert14">alert14</a>
<a href="#gdcalert15">alert15</a>
<a href="#gdcalert16">alert16</a>
<a href="#gdcalert17">alert17</a>
<a href="#gdcalert18">alert18</a>
<a href="#gdcalert19">alert19</a>
<a href="#gdcalert20">alert20</a>
<a href="#gdcalert21">alert21</a>
<a href="#gdcalert22">alert22</a>
<a href="#gdcalert23">alert23</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>



## Setup an Azure Machine Learning Workspace (30-35%)

Training and deploying an effective machine learning model involves a lot of work, much of it time-consuming and resource-intensive. Azure Machine Learning is a cloud-based service that helps simplify some of the tasks and reduce the time it takes to prepare data, train a model, and deploy a predictive service.



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")




<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image2.png "image_tooltip")


The diagram shows the following components of a workspace:



*   
A workspace can contain Azure Machine Learning compute instances, cloud resources configured with the Python environment necessary to run Azure Machine Learning.


*   
User roles enable you to share your workspace with other users, teams, or projects.


*   
Compute targets are used to run your experiments.


*   
When you create the workspace, associated resources are also created for you.


*   
Experiments are training runs you use to build your models.


*   
Pipelines are reusable workflows for training and retraining your model.


*   
Datasets aid in management of the data you use for model training and pipeline creation.


*   
Once you have a model you want to deploy, you create a registered model.


*   
Use the registered model and a scoring script to create a deployment endpoint.


*   
Create an Azure ML Workspace
There are four ways to create a workspace:

	1- Azure Portal

	2- Python SDK - Jupyter or IDE

	3- Azure Resource Manager Template or Azure ML CLI


    4- Visual Studio Code Extension

With Python you can connect to a workspace using a config file, and also delete workspaces

Select ＋Create a resource, search for Machine Learning, and create a new Machine Learning resource the following settings:



*   **Subscription**: Your Azure subscription
*   **Resource group**: Create or select a resource group
*   **Workspace name**: Enter a unique name for your workspace
*   **Region**: Select the geographical region closest to you
*   **Storage account**: Note the default new storage account that will be created for your workspace
*   **Key vault**: Note the default new key vault that will be created for your workspace
*   **Application insights:** Note the default new application insights resource that will be created for your workspace
*   **Container registry:** None (one will be created automatically the first time you deploy a model to a container)

On the overview page, launch the Azure Machine Learning Studio and sign in with the microsoft account. Then, create  computing resources: **Computing Instances and Clusters. **Suggestions from MFST:

Computing Instance Config:



*   Virtual Machine type: CPU
*   Virtual Machine size:
    *   Choose Select from all options
    *   Search for and select Standard_DS11_v2
*   Compute name: enter a unique name
*   Enable SSH access: Unselected

Cluster Config:



*   Virtual Machine type: CPU
*   Virtual Machine size:
    *   Choose Select from all options
    *   Search for and select Standard_DS11_v2
*   Compute name: enter a unique name
*   Minimum number of nodes: 0
*   Maximum number of nodes: 2
*   Idle seconds before scale down: 120

With Resources created, you are free to load data, explore and use Jupyter Notebooks or Rstudio.



*   
Run an Automated Machine Learning Experiment
**From the Azure ML Studio, open the Automated ML page, and click “Create New** Automated ML run”



*   Select dataset:
    *   Dataset: bike-rentals
*   Configure run:
    *   New experiment name: mslearn-bike-rental
    *   Target column: rentals (this is the label the model will be trained to predict)
    *   Select compute cluster: the compute cluster you created previously
*   Task type and settings:
    *   Task type: Regression (the model will predict a numeric value)
    *   Additional configuration settings:
        *   Primary metric: Select Normalized root mean squared error (more about this metric later!)
        *   Explain best model: Selected - this option causes automated machine learning to calculate feature importance for the best model; making it possible to determine the influence of each feature on the predicted label.
        *   Blocked algorithms: Block all other than RandomForest and LightGBM - normally you'd want to try as many as possible, but doing so can take a long time!
        *   Exit criterion:
            *   Training job time (hours): 0.5 - this causes the experiment to end after a maximum of 30 minutes.
            *   Metric score threshold: 0.08 - this causes the experiment to end if a model achieves a normalized root mean squared error metric score of 0.08 or less.
*   Featurization settings:
    *   Enable featurization: Selected - this causes Azure Machine Learning to automatically preprocess the features before training.



*   
Review the best model
On the Details tab of the automated machine learning run, note the best model summary.

Select the Algorithm name for the best model to view its details.

The best model is identified based on the evaluation metric you specified (Normalized root mean squared error). To calculate this metric, the training process used some of the data to train the model, and applied a technique called cross-validation to iteratively test the trained model with data it wasn't trained with and compare the predicted value with the actual known value. The difference between the predicted and actual value (known as the residuals) indicates the amount of error in the model, and this particular performance metric is calculated by squaring the errors across all of the test cases, finding the mean of these squares, and then taking the square root. What all of this means is that smaller this value is, the more accurately the model is predicting.

Next to the Normalized root mean squared error value, select View all other metrics to see values of other possible evaluation metrics for a regression model.

Select the Metrics tab and select the residuals and predicted_true charts if they are not already selected. Then review the charts, which show the performance of the model by comparing the predicted values against the true values, and by showing the residuals (differences between predicted and actual values) as a histogram.

The Predicted vs. True chart should show a diagonal trend in which the predicted value correlates closely to the true value. A dotted line shows how a perfect model should perform, and the closer the line for your model's average predicted value is to this, the better its performance. A histogram below the line chart shows the distribution of true values.



*   
Deploy Model as a Service
In Azure Machine Learning, you can deploy a service as an Azure Container Instances (ACI) or to an Azure Kubernetes Service (AKS) cluster. For production scenarios, an AKS deployment is recommended, for which you must create an inference cluster compute target. In this exercise, you'll use an ACI service, which is a suitable deployment target for testing, and does not require you to create an inference cluster.



*   In Azure Machine Learning studio, on the Automated ML page, select the run for your automated machine learning experiment and view the Details tab.
*   Select the algorithm name for the best model. Then, on the Model tab, use the Deploy button to deploy the model with the following settings:
    *   Name: predict-rentals
    *   Description: Predict cycle rentals
    *   Compute type: Azure Container Instance
    *   Enable authentication: Selected
*   Wait for the deployment to start - this may take a few seconds. Then, in the Model summary section, observe the Deploy status for the predict-rentals service, which should be Running. Wait for this status to change to Successful. You may need to select ↻ Refresh periodically.
*   In Azure Machine Learning studio, view the Endpoints page and select the predict-rentals real-time endpoint. Then select the Consume tab and note the following information there. You need this information to connect to your deployed service from a client application.
    *   The REST endpoint for your service
    *   **the** Primary Key for your service



*   
Manage data objects in an Azure Machine Learning workspace

#### Data workflow

When you're ready to use the data in your cloud-based storage solution, we recommend the following data delivery workflow. This workflow assumes you have an Azure storage account and **data** in a cloud-based storage service in Azure.



*   Create an Azure Machine Learning datastore to store connection information to your Azure storage.
*   From that datastore, create an Azure Machine Learning dataset to point to a specific file(s) in your underlying storage.
*   To use that dataset in your machine learning experiment you can either
    *   Mount it to your experiment's compute target for model training.
*   OR
    *   Consume it directly in Azure Machine Learning solutions like, automated machine learning (automated ML) experiment runs, machine learning pipelines, or the Azure Machine Learning designer.
*   Create dataset monitors for your model output dataset to detect for data drift.
*   If data drift is detected, update your input dataset and retrain your model accordingly.



<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image3.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image3.png "image_tooltip")


There are two types of dataset:



*   **File dataset** - references single or multiple files in your datastores or public URLs. If your data is already cleansed and ready to use in training experiments, you can download or mount files referenced by FileDatasets to your compute target.
*   **A Tabular Dataset** represents data in a tabular format by parsing the provided file or list of files. You can load a TabularDataset into a pandas or Spark DataFrame for further manipulation and cleansing. For a complete list of data formats you can create TabularDatasets from, see the TabularDatasetFactory class.



*   
Manage experiment compute contexts


<p id="gdcalert4" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image4.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert5">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image4.png "image_tooltip")



#### Compute Instance Vs Compute Cluster



<p id="gdcalert5" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image5.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert6">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image5.png "image_tooltip")



#### Attached compute

To use compute targets created outside the Azure Machine Learning workspace, you must attach them. Attaching a compute target makes it available to your workspace. Use Attached compute to attach a compute target for training. Use Inference clusters to attach an AKS cluster for inferencing. It can be:



*   An Azure Virtual Machine (to attach a Data Science Virtual Machine)
*   Azure Databricks (for use in machine learning pipelines)
*   Azure Data Lake Analytics (for use in machine learning pipelines)
*   Azure HDInsight
*   Kubernetes (preview)


## Run experiments and train models (25–30%)


### Create models by using Azure ML designer

The designer gives you a visual canvas to build, test, and deploy machine learning models. With the designer you can:



*   Drag-and-drop datasets and modules onto the canvas.
*   Connect the modules to create a pipeline draft.
*   Submit a pipeline run using the compute resources in your Azure Machine Learning workspace.
*   Convert your training pipelines to inference pipelines.
*   Publish your pipelines to a REST pipeline endpoint to submit a new pipeline that runs with different parameters and datasets.
    *   Publish a training pipeline to reuse a single pipeline to train multiple models while changing parameters and datasets.
    *   Publish a batch inference pipeline to make predictions on new data by using a previously trained model.
*   Deploy a real-time inference pipeline to a real-time endpoint to make predictions on new data in real time.

A valid pipeline has these characteristics:



*   Datasets can only connect to modules.
*   Modules can only connect to either datasets or other modules.
*   All input ports for modules must have some connection to the data flow.
*   All required parameters for each module must be set.
*   For real time publishing, kubernetes is mandatory



<p id="gdcalert6" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image6.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert7">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image6.png "image_tooltip")




<p id="gdcalert7" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image7.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert8">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image7.png "image_tooltip")



### Understand what happens when you submit a training job

The Azure training lifecycle consists of:



*   Zipping the files in your project folder, ignoring those specified in .amlignore or .gitignore
*   Scaling up your compute cluster
*   Building or downloading the dockerfile to the compute node
    *   The system calculates a hash of:
        *   The base image
        *   Custom docker steps (see Deploy a model using a custom Docker base image)
        *   The conda definition YAML (see Create & use software environments in Azure Machine Learning)
    *   The system uses this hash as the key in a lookup of the workspace Azure Container Registry (ACR)
    *   If it is not found, it looks for a match in the global ACR
    *   If it is not found, the system builds a new image (which will be cached and registered with the workspace ACR)
*   Downloading your zipped project file to temporary storage on the compute node
*   Unzipping the project file
*   The compute node executing python &lt;entry script> &lt;arguments>
*   Saving logs, model files, and other files written to ./outputs to the storage account associated with the workspace
*   Scaling down compute, including removing temporary storage
*   If you choose to train on your local machine ("configure as local run"), you do not need to use Docker. You may use Docker locally if you choose (see the section Configure ML pipeline for an example).


### Generate metrics from an experiment run


#### Enable Logging with Execute Python Script



<p id="gdcalert8" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image8.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert9">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image8.png "image_tooltip")



#### View Logs

After the pipeline run completes, you can see the Mean_Absolute_Error in the Experiments page.



*   Navigate to the Experiments section.
*   Select your experiment.
*   Select the run in your experiment you want to view.
*   Select Metrics.



<p id="gdcalert9" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image9.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert10">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image9.png "image_tooltip")



### Automate the model training process

Common kinds of step in an Azure Machine Learning pipeline include:



*   PythonScriptStep: Runs a specified Python script.
*   DataTransferStep: Uses Azure Data Factory to copy data between data stores.
*   DatabricksStep: Runs a notebook, script, or compiled JAR on a databricks cluster.
*   AdlaStep: Runs a U-SQL job in Azure Data Lake Analytics.
*   ParallelRunStep - Runs a Python script as a distributed task on multiple compute nodes.



<p id="gdcalert10" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image10.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert11">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image10.png "image_tooltip")



## Optimize and manage models (20–25%)


### Automate the model training process



<p id="gdcalert11" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image11.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert12">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image11.png "image_tooltip")




<p id="gdcalert12" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image12.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert13">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image12.png "image_tooltip")



### Use HyperDrive to tune hyperparameters


#### Defining a search space

To define a search space for hyperparameter tuning, create a dictionary with the appropriate parameter expression for each named hyperparameter. For example, the following search space indicates that the batch_size hyperparameter can have the value 16, 32, or 64, and the learning_rate hyperparameter can have any value from a normal distribution with a mean of 10 and a standard deviation of 3.



<p id="gdcalert13" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image13.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert14">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image13.png "image_tooltip")



#### Sampling:



*   Grid Sampling - For discrete hyperparameters, trying all combinations
*   Random Sampling
*   Bayesian Sampling - Use Bayesian stats to optimise search 


#### Early Termination:



*   Bandit Policy - Stop a run if the target performance metric underperforms the best run so far by a specific margin
*   Median Stop Policy - Abandons runs where the target performance metric is worse than the median of the running averages for all runs.
*   Truncation Selection Policy - Cancels the lowest performing X% of runs at each evaluation interval based on the truncation_percentage value you specify for X.

Example Script for Parameter Tuning



<p id="gdcalert14" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image14.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert15">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image14.png "image_tooltip")




<p id="gdcalert15" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image15.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert16">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image15.png "image_tooltip")



### Use Model Explainers to interpret Models


#### How to Interpret your model:

Using the classes and methods in the SDK, you can:



*   Explain model prediction by generating feature importance values for the entire model and/or individual datapoints.
*   Achieve model interpretability on real-world datasets at scale, during training and inference.
*   Use an interactive visualization dashboard to discover patterns in data and explanations at training time
*   

In machine learning, features are the data fields used to predict a target data point. For example, to predict credit risk, data fields for age, account size, and account age might be used. In this case, age, account size, and account age are features. Feature importance tells you how each data field affected the model's predictions. For example, age may be heavily used in the prediction while account size and age do not affect the prediction values significantly. This process allows data scientists to explain resulting predictions, so that stakeholders have visibility into what features are most important in the model.



<p id="gdcalert16" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image16.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert17">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image16.png "image_tooltip")




<p id="gdcalert17" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image17.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert18">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image17.png "image_tooltip")



#### Feature Importance:

Feature values are randomly shuffled, one column at a time. The performance of the model is measured before and after. You can choose one of the standard metrics to measure performance.

The scores that the module returns represent the change in the performance of a trained model, after permutation. Important features are usually more sensitive to the shuffling process, so they'll result in higher importance scores.

On Azure ML:



*   Add the Permutation Feature Importance module to your pipeline. You can find this module in the Feature Selection category.
*   Connect a trained model to the left input. The model must be a regression model or a classification model.
*   On the right input, connect a dataset. Preferably, choose one that's different from the dataset that you used for training the model. This dataset is used for scoring based on the trained model. It's also used for evaluating the model after feature values have changed.
*   For Random seed, enter a value to use as a seed for randomization. If you specify 0 (the default), a number is generated based on the system clock.
*   A seed value is optional, but you should provide a value if you want reproducibility across runs of the same pipeline.
*   For Metric for measuring performance, select a single metric to use when you're computing model quality after permutation.
*   Azure Machine Learning designer supports the following metrics, depending on whether you're evaluating a classification or regression model:
    *   Classification
        *   Accuracy, Precision, Recall
    *   Regression
        *   Precision, Recall, Mean Absolute Error, Root Mean Squared Error, Relative Absolute Error, Relative Squared Error, Coefficient of Determination
*   For a more detailed description of these evaluation metrics and how they're calculated, see Evaluate Model.
*   Submit the pipeline.
*   The module outputs a list of feature columns and the scores associated with them. The list is ranked in descending order of the scores.


#### Machine Learning Fairness

Artificial intelligence and machine learning systems can display unfair behavior. One way to define unfair behavior is by its harm, or impact on people. There are many types of harm that AI systems can give rise to.

Two common types of AI-caused harms are:



*   Harm of allocation: An AI system extends or withholds opportunities, resources, or information for certain groups. Examples include hiring, school admissions, and lending where a model might be much better at picking good candidates among a specific group of people than among other groups.
*   Harm of quality-of-service: An AI system does not work as well for one group of people as it does for another. As an example, a voice recognition system might fail to work as well for women as it does for men.

In the Fairlearn open-source package, fairness is conceptualized through an approach known as group fairness, which asks: Which groups of individuals are at risk for experiencing harm? The relevant groups, also known as subpopulations, are defined through sensitive features or sensitive attributes. Sensitive features are passed to an estimator in the Fairlearn open-source package as a vector or a matrix called sensitive_features. The term suggests that the system designer should be sensitive to these features when assessing group fairness.

Mitigating unfairness in a model means reducing the unfairness, but this technical mitigation cannot eliminate this unfairness completely. The unfairness mitigation algorithms in the Fairlearn open-source package can provide suggested mitigation strategies to help reduce unfairness in a machine learning model, but they are not solutions to eliminate unfairness completely. There may be other parity constraints or criteria that should be considered for each particular developer's machine learning model. Developers using Azure Machine Learning must determine for themselves if the mitigation sufficiently eliminates any unfairness in their intended use and deployment of machine learning models.



<p id="gdcalert18" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image18.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert19">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image18.png "image_tooltip")




<p id="gdcalert19" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image19.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert20">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image19.png "image_tooltip")



### Manage Models

The workflow is similar no matter where you deploy your model:



*   Register the model
*   Prepare an entry script
*   Prepare an inference configuration
*   Deploy the model locally to ensure everything works
*   Choose a compute target
*   Re-deploy the model to the cloud
*   Test the resulting web service



<p id="gdcalert20" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image20.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert21">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image20.png "image_tooltip")




<p id="gdcalert21" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image21.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert22">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image21.png "image_tooltip")



#### Dataset Monitors

With a dataset monitor you can:



*   Detect and alert to data drift on new data in a dataset.
*   Analyze historical data for drift.
*   Profile new data over time.
*   The data drift algorithm provides an overall measure of change in data and indication of which features are responsible for further investigation. Dataset monitors produce a number of other metrics by profiling new data in the timeseries dataset.

Custom alerting can be set up on all metrics generated by the monitor through Azure Application Insights. Dataset monitors can be used to quickly catch data issues and reduce the time to debug the issue by identifying likely causes.

Conceptually, there are three primary scenarios for setting up dataset monitors in Azure Machine Learning



<p id="gdcalert22" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image22.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert23">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image22.png "image_tooltip")




<p id="gdcalert23" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image23.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert24">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image23.png "image_tooltip")



## Deploy and consume models (20–25%)
