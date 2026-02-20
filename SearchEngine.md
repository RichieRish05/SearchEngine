Search Engine Specification
Search engine
This assignment is to be done in groups of 1, 2, 3 or 4. You can work on the same groups that were in place for the crawler project if you wish to. Although this is presented as one single project here, internally it is organized in 3 separate milestones, each with a specific deadline, deliverables and score. You can only change groups between the milestones if there are strong reasons for doing so. Please, write an email to the teaching team, explaining why you need to change teams, and we will assess your situation. 

In doing milestones #1 and #2, make sure to consider the evaluation criteria not just of those milestones but also of milestone #3 — part of the milestones’ evaluation will be delayed until the final meeting with the TAs. You can use code that you or any classmate wrote for the previous projects.
You cannot use code written for this project by non-group-member classmates. You are allowed to use any languages and libraries you want for text processing, including nltk. However, you are not allowed to use text indexing libraries such as Lucene, PyLucene, or ElasticSearch.

To accommodate the various skill levels of students in this course, this assignment comes in two flavors:

Information Analyst. In this flavor there is some programming involved, but not much more advanced than what you already did so far.
It is a mixture of a Text Processing project and stitching things together. You will use a small subset of crawled pages. Groups where ALL students are NOT CS nor SE majors can choose this.
Algorithms and Data Structures Developer. In this flavor there is more programming, and your code needs to perform well on the
entire collection of crawled pages, under the required constraints. This option is available to everyone, but groups that have at least one
CS or SE student are required to do this.
Milestones Overview
Milestone	Goal	Due date	Deliverables	Score
1	Initial index	See Assignment 3: M1	Short report + code	5
2	Retrieval component	See Assignment 3: M2	Short report + code	5
3	Complete search system	See Assignment 3: M3	Report + code + demo	50
General Specifications
You will develop two separate programs: an indexer and a search component.

Indexer:
Create an inverted index for the given corpus with data structures designed by you.

Tokens: all alphanumeric sequences in the dataset.
Stop words: do not use stopping, i.e. use all words, even the frequently occurring ones.
Stemming: use stemming for better textual matches. Suggestion: Porter stemming.
Important words: Words in bold, in headings (h1, h2, h3), and in titles should be treated as more important than the other words.
Search:
Your program should prompt the user for a query. This doesn’t need to be a Web interface, it can be a console prompt. At the time of the query, your program will stem the query terms, look up your index, perform some calculations (see ranking below) and give out the ranked list of pages that are relevant for the query, with the most relevant on top. Pages should be identified by their URLs.

Ranking: at the very least, your ranking formula should include tf-idf scoring, and take the important words into consideration, but you should feel free to add additional components to this formula if you think they improve the retrieval. 

Extra Credit:
Extra credit will be given for tasks that improve the quality of the retrieval and the of the search experience. For example:

Detect and eliminate duplicate pages. (1 point for exact, 2 points for near)
Add HITS and/or Page Rank to your ranking formula. (1.5 for HITS, 2.5 for PR)
Implement an additional 2-gram and/or 3-gram indexing and use it during retrieval. (1 point)
Enhance the index with word positions and use that information for retrieval. (2 points)
Index anchor words for the target pages (1 point).
Implement a Web or GUI interface instead of a console one. (1 point for local GUI, 2 points for Web interface)
 

Additional Specifications for Information Analyst
Option available to all groups formed only by non-CS and non-SE students.
Programming skills required: Intro courses
Main challenges: HTML and JSON parsing, read/write structured information from/to files or databases.
Corpus: a small portion of the ICS web pages (analyst.zip)
Indexer:

You can use a database to store the index, or a simple file – whatever is simpler to you. If you store it in a file, the index is expected to be sufficiently small, so that it fits in memory all at once.

Search interface:

The response to search queries should be less than 2 seconds.

Note:
This project is a great addition to your résumé!
Tired: “Wrote a Python script that finds words in Web pages.”
Wired: “Wrote a search engine from the ground up that is capable of handling two thousand Web pages.”

 

Additional Specifications for Algorithms and Data Structures Developer
Option available to all students, but required for CS and SE students.
Programming skills required: advanced
Main challenges: design efficient data structures, devise efficient file access, balance memory usage and response time
Corpus: all ICS web pages (developer.zip)
Index: Your index should be stored in one or more files in the file system (no databases!).
Search interface:

The response to search queries should be 
 300ms. Ideally, it would be 
 100ms, or less, but you won’t be penalized if it’s higher (as long as it’s kept 
 300ms).

Operational constraints: 

Typically, the cloud servers/containers that run search engines don’t have a lot of memory, but they need to handle large amounts of data. As such, you must design and implement your programs as if you are dealing with very large amounts of data, so large that you cannot hold the inverted index all in memory. Your indexer must offload the inverted index hash map from main memory to a partial index on disk at least 3 times during index construction; those partial indexes should be merged in the end. Optionally, after or during merging, they can also be split into separate index files with term ranges. similarly, your search component must not load the entire inverted index in main memory. Instead, it must read the postings from the index(es) files on disk. The TAs will check that both of these things are happening.

Note:
This project is a great addition to your résumé!
Tired: “Wrote a Web search engine using ElasticSearch.”
Wired: “Wrote a Web search engine from the ground up that is capable of handling tens of thousands of Web pages, under harsh operational constraints and having a query response time under 300ms.”

Data sources:
(1) Information analyst: small collection of web pages

(2) Algorithms and data structures developer: larger collection of web pages

Understanding the Dataset

Your crawlers crawled the many web sites associated with ICS. We collected a big chunk of these pages and are providing them to you as two zip files: analyst.zip and developer.zip. The names are self-explanatory: the former is for the analyst flavor of the project; the latter is for the algorithms and data structures developer option. The only difference between them is the size: analyst.zip contains only 3 sub-domains and a little over 2,000 pages, while developer.zip contains all 88 sub-domains found during crawling and a little under 56,000 pages.

The following is an explanation of how the data is organized.

Folders:

There is one folder per domain. Each file inside a folder corresponds to one web page. (note that you would not do this in a real search engine). 

Files:

The files are stored in JSON format, with the following fields :

“url” : contains the URL of the page. (ignore the fragment part, if you see it)
“content” : contains the content of the page, as found during crawling
"encoding" : an indication of the encoding of the webpage
Broken or missing HTML:

Real HTML pages found out there are full of bugs! Some of the pages in the dataset may not contain any HTML at all and, when they do, it may not be well formed. For example, there might be an open <strong> tag but the associated closing </strong> tag might be missing. While selecting the parser library for your project, please ensure that it can handle broken HTML.