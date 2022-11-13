# **Data extraction using NLP & spaCy in Resume Parsing**

Using spaCy to train and test a resume parsing model

![image](https://user-images.githubusercontent.com/72379583/201525319-7c22e5f9-83f8-4053-94fa-6d279b5447af.png)

# Introduction:
Every day, corporations and recruiting firms must process numerous amount of resumes. Working with large amounts of text data, which can be time-consuming and stressful. Data collected from various resumes can be in a variety of formats, such as PDF, DOCX, single-column resumes, double-column resumes, free formats, and so on. And these formats may not be appropriate for the specific application. As a result, we may wonder, “What is resume parsing?” Resume parsing is the process of converting unstructured resume data (.pdf/.docx/.jpeg, etc.) into a structured format.

Sounds easy right? At first, I assumed it was quite simple. Just use some patterns to mine the data, but it turns out that I was wrong! Building a resume parser is difficult because there are so many types of resume layouts that you could imagine.

Let’s have a look at some essential topics to get a better understanding of how resume parsers work.

## spaCy

              spaCy is a popular and easy-to-use natural language processing library in Python. It provides current state-of-the-art accuracy and speed levels, and has an active open source community.

spaCy enables you to create applications that process and “understand” large amounts of text. It can be used to create systems for information extraction, NLU, or text pre-processing before deep learning.

## train_test_split

                  The **`train_test_split`** function of the **`sklearn.model_selection`** package in Python splits arrays or matrices into random subsets for train and test data, respectively.

To use the `train_test_split` function, we’ll import it into our program as shown below:

```
from sklearn.model_selection import train_test_split
train, test = train_test_split(cv_data, test_size=0.3)
```

## PyMuPDF

                  It is a simple Python package for text-based information extraction from PDF files that can be used to test our model.
                  
# Implementation:

# Creating a Resume Parser Using spaCy

                          Using SpaCy, we will build a model that will extract the key points from a resume. We plan to train the model on nearly 200 resumes. When the model is complete, we will extract the text from a new resume and feed it into the model to generate the summary.

## Step #1: Setting up Google Colab

To train our model, We need more GPU-oriented processing, which we can also do in our local system with the help of Jupyter. I chose to work in colab notebook because it provides free access to computing resources, including GPUs, and it is simple to set up the GPU.

Also, I would like to run the project on my local system, and I tried it, but creating a GPU environment for my Jupiter notebook didn't go well for various reasons, so the catch here is that if you can configure your local pc, go for it because colab has some GPU usage limits, and if I succeed in accessing my local GPU via Jupyter, I will certainly create a blog on that.

![image](https://user-images.githubusercontent.com/72379583/201525433-643f70db-1166-4367-a5f4-4b801d44408e.png)

So, now that you've successfully created your Colab notebook, proceed to package installation and import..

## Step #2: Package installation and import

Now we need to import the required packages, which we discussed earlier; however, before we do so, we must pip install them into our environment to ensure that the code continues to function properly even after months and to avoid changes in function names in the future. I have mentioned the versions of spaCy and spaCy transformers. 

```
!pip install spacy_transformers==1.1.4
!pip install spacy==3.3.1
```

Since we had no trouble installing the packages, it is now time to import them. Import the following packages, and if they are missing, simply pip install them.

```
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json
```

Moving on to Data Preparation

## Step #3: Data Preparation

When building any machine learning model, collecting training data can be a time-consuming and frustrating experience. It may appear to be a nightmarish procedure. In this project, we used approximately 200 resumes to train our model.

![image](https://user-images.githubusercontent.com/72379583/201525481-42055fac-318e-4040-8093-506b1140f687.png)

So, for this project, I used training data from Kaggle, which made it easier for me to collect the data. In most cases, you won't be able to find your desired data set online, so you'll have to create it yourself. How should it be done? Hmm, just look it up, and you'll get an idea. You can also refer to the article below; it helped me understand data preparation better.

[Data Preparation in Machine Learning: 6 Key Steps](https://www.techtarget.com/searchbusinessanalytics/feature/Data-preparation-in-machine-learning-6-key-steps)

Now, it’s time for me to explain about training data

```jsx
['Govardhana K Senior Software Engineer  Bengaluru, Karnataka, Karnataka - Email me on Indeed: indeed.com/r/Govardhana-K/ b2de315d95905b68  Total IT experience 5 Years 6 Months Cloud Lending Solutions INC 4 Month • Salesforce Developer Oracle 5 Years 2 Month • Core Java Developer Languages Core Java, Go Lang Oracle PL-SQL programming, Sales Force Developer with APEX.  Designations & Promotions  Willing to relocate: Anywhere  WORK EXPERIENCE  Senior Software Engineer  Cloud Lending Solutions -  Bangalore, Karnataka -  January 2018 to Present  Present  Senior Consultant  Oracle -  Bangalore, Karnataka -  November 2016 to December 2017  Staff Consultant  Oracle -  Bangalore, Karnataka -  January 2014 to October 2016  Associate Consultant  Oracle -  Bangalore, Karnataka -  November 2012 to December 2013  EDUCATION  B.E in Computer Science Engineering  Adithya Institute of Technology -  Tamil Nadu  September 2008 to June 2012  https://www.indeed.com/r/Govardhana-K/b2de315d95905b68?isid=rex-download&ikw=download-top&co=IN https://www.indeed.com/r/Govardhana-K/b2de315d95905b68?isid=rex-download&ikw=download-top&co=IN   SKILLS  APEX. (Less than 1 year), Data Structures (3 years), FLEXCUBE (5 years), Oracle (5 years), Algorithms (3 years)  LINKS  https://www.linkedin.com/in/govardhana-k-61024944/  ADDITIONAL INFORMATION  Technical Proficiency:  Languages: Core Java, Go Lang, Data Structures & Algorithms, Oracle PL-SQL programming, Sales Force with APEX. Tools: RADTool, Jdeveloper, NetBeans, Eclipse, SQL developer, PL/SQL Developer, WinSCP, Putty Web Technologies: JavaScript, XML, HTML, Webservice  Operating Systems: Linux, Windows Version control system SVN & Git-Hub Databases: Oracle Middleware: Web logic, OC4J Product FLEXCUBE: Oracle FLEXCUBE Versions 10.x, 11.x and 12.x  https://www.linkedin.com/in/govardhana-k-61024944/',
 {'entities': [[1749, 1755, 'Companies worked at'],
   [1696, 1702, 'Companies worked at'],
   [1417, 1423, 'Companies worked at'],
   [1356, 1793, 'Skills'],
   [1209, 1215, 'Companies worked at'],
   [1136, 1247, 'Skills'],
   [928, 932, 'Graduation Year'],
   [858, 889, 'College Name'],
   [821, 856, 'Degree'],
   [787, 791, 'Graduation Year'],
   [744, 750, 'Companies worked at'],
   [722, 742, 'Designation'],
   [658, 664, 'Companies worked at'],
   [640, 656, 'Designation'],
   [574, 580, 'Companies worked at'],
   [555, 572, 'Designation'],
   [470, 493, 'Companies worked at'],
   [444, 468, 'Designation'],
   [308, 314, 'Companies worked at'],
   [234, 240, 'Companies worked at'],
   [175, 198, 'Companies worked at'],
   [93, 136, 'Email Address'],
   [39, 48, 'Location'],
   [13, 37, 'Designation'],
   [0, 12, 'Name']]}]
```

So, what you're looking at is one of 200 labelled resumes that we used to train our model,`[0, 12, 'Name']` Consequently,  this is how labelled data looks. You can see the index numbers mentioned before the entities (name, skills) so that SpaCy understands what is provided in that section of the data.

Furthermore, read the article below to learn more about data preparation and to get a clear understanding of data labelling.

[What is Data Labeling? Everything You Need To Know With Meeta Dash](https://appen.com/blog/data-labeling/)

coming up next step#4

## Step #4: Model Training & Testing

 
![image](https://user-images.githubusercontent.com/72379583/201525505-506c2e38-879a-421a-a61c-ecfa676a0221.png)

Training data for an NLP project may be found in a variety of formats. SpaCy provides converters for some popular formats, such as CoNLL. In other situations, you must prepare the training data yourself.

When converting training data for use in spaCy, the main thing to remember is to create Doc objects that look exactly like the results you want as pipeline output. On disk the annotations will be saved as a `[DocBin](https://spacy.io/api/docbin)`in the `[.spacy`](https://spacy.io/api/data-formats#binary-training) format, but the details of that are handled automatically.

We're making a `[.spacy](https://spacy.io/api/data-formats#binary-training)` file out of some NER annotations, I'm not displaying the code for that process here because it's quite lengthy; instead, I've provided a link to the code  at the end of this blog you can see there.

```jsx
file = open('error.txt', 'w')

db = get_spacy_docs(file, train)
db.to_disk('train_data.spacy')

db = get_spacy_docs(file, test)
db.to_disk('test_data.spacy')

file.close()
```

```jsx

```

We can now train our model because we have created test data and train data in a format that spacy understands.

```jsx
!python -m spacy train /content/drive/MyDrive/Projects/--Res_parser/CV-Parsing-using-Spacy-3-master/data/training/config.cfg --output ./output --paths.train /content/drive/MyDrive/Projects/--Res_parser/CV-Parsing-using-Spacy-3-master/data/training/train_data.spacy --paths.dev /content/drive/MyDrive/Projects/--Res_parser/CV-Parsing-using-Spacy-3-master/data/training/test_data.spacy --gpu-id 0
```

You can now add your data and run train with your config. It will take a long time to train the model. As a result, we are storing the model for future usage.

Check out this article to know more about model training which helped me to build this project.

[Training Pipelines & Models · spaCy Usage Documentation](https://spacy.io/usage/training)

Now that we have successfully trained our model, we can put it to the test.

```jsx
fname = "Rajkumar.pdf"
doc = fitz.open(fname)

text = " "
for page in doc:
  text = text + str(page.get_text())
```

Now we'll send this resume to our model and see what happens. We printed with some formatting.

```jsx
doc = nlp(text)
for ent in doc.ents:
  print(ent.text, " ->>>>>> ", ent.label_)
```

We are currently testing our model on an unseen resume. Because the resume is in PDF format, we will use `[PyMuPDF](https://pypi.org/project/PyMuPDF/)` to extract the text from the PDF file. Then we'll feed the text into our model and see what happens.

```python
NAME ->>>>>> RajKumar
LOCATION ->>>>>> Delhi
DESIGNATION ->>>>>> Stream Analytics
DESIGNATION ->>>>>> Software Engineer
COMPANIES WORKED AT ->>>>>> Microsoft –
DEGREE ->>>>>> Indian Institute of Technology – Mumbai
SKILLS ->>>>>> Machine Learning, Natural Language Processing, and Big Data Handling    ADDITIONAL INFORMATION  Professional Skills  • Excellent analytical, problem solving, communication, knowledge transfer and interpersonal
```

Our project's output can be seen here. You can train the model on more data samples to get a more accurate summary. In the training samples, you can include various types of resumes.

# Challenges

- It took some time for me to grasp the objective. ****
- I did not plan ahead of time for this project. Now I’m regretting that.
- I got stuck at some points while implementing, and it took awhile to find a solution.

# Pros & Cons

Some clear and straightforward advantages

1. Resume parsing can save hiring managers hours spent manually reading through each resume and organizing those with relevant skills and information and eliminating those without.
2. Using a resume parser increases your chances of finding a variety of qualified candidates who match the job descriptions of open positions at your company. 
3. With technological advancements, we can now parse a candidate's social media page, such as their LinkedIn page, into a usable format.

There are some downsides that you’ll want to keep in mind too

1. When using a resume parser, it is possible to overlook a highly qualified candidate. The ideal candidate may fall through the cracks.
2. With the “right” keywords, resume parsing can be manipulated, allowing candidates to appear to be the better fit for the job.
3. This project has room for improvement and optimization.
