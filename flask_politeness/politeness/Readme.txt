Stanford Politeness Corpus v1.01 (released May 2013)

Distributed together with:

"A computational approach to politeness with application to social factors"
Cristian Danescu-Niculescu-Mizil, Moritz Sudhof, Dan Jurafsky, Jure Leskovec, Christopher Potts
Proceedings of ACL, 2013.

NOTE: If you have results to report on this corpus, please send an email to cristiand@cs.stanford.edu so we can add you to our list of people using this data.  Thanks!



Contents of this README:

	A) Brief description
	B) Files description
	C) Experimental setting
	D) Contact


A) Brief description:

This corpus contains a collection of requests from Wikipedia editor's talk pages (http://en.wikipedia.org/wiki/Wikipedia:Talk_page_guidelines) and from the StackExchange question-answering communities (http://stackexchange.com/about) partially annotated for politeness.  There are 35,661 requests from Wikipedia out of which 4,353 are annotated and 373,519 requests from StackExchange out of which 6,603 are annotated.  Metadata includes: 

- timestamps (Wikipedia and StackExchange)
- editor's status (Wikipedia)
- user's reputation (StackExchange)
- up/down votes (StackExchange)


Requests are in English. Details about the annotation process and about the data collection are described in Section 2 of the paper referenced above.


B) Files description

All files are in csv format (using ',' as the delimiter and '"' as the quoting character); the fields are indicated in the header of each file.  Note that entries may contain line breaks (in which case, the quoting character is used).


<===> wikipedia.requests.csv

Requests extracted from Wikipedia talkpages.  The fields are:

Community - always "Wikipedia" in this file
Id - unique ID of the request
Request - text of the request
Timestamp - time when the request was posted ("NA" if not known)
User - Wikipedia user name


-----------------------------------


<===> wikipedia.annotated.csv


Subset of the Wikipedia requests for which we collected politeness annotations. The fields are:

Community - always "Wikipedia" in this file
Id - Unique ID of the request (can be used to match the requests with the metadata in  wikipedia.requests.csv)
Request - text of the request
Score[1-5] - politeness scores assigned by each of the 5 annotators, between 1 (very impolite) and 25 (very polite)
TurkId[1-5]	- the Amazon Mechanical Turk ids of each of the 5 annotators
Normalized Score - the normalized politeness score of the requests (after normalizing each annotator's score; details in Section 2 of the paper)


-----------------------------------

<===>  wikipedia.admins.csv

List of users in our data with administrator status (http://en.wikipedia.org/wiki/Wikipedia:Administrators) at the time this data was collected.  The date when this status was gained through a Request for Adminship election process (http://en.wikipedia.org/wiki/Wikipedia:Requests_for_adminship) is indicated in yyyy-mm-dd  format (missing dates are indicated with NA)

-----------------------------------

<===> stack-exchange.requests.csv

Requests extracted from StackExchange comments.  The fields are:

Community - the StackExchange community in which the request was posted (e.g., "092011 Cooking" corresponds comments to posts from the the cooking StackExchange site http://cooking.stackexchange.com/)
Id - ID of the request, unique withing the community
Request - text of the request
Timestamp - time when the request was posted
UserId - StackExchange user id (unique within the community)
Reputation - reputation score (at the time of the data collection)
Upvotes - number of upvotes the user received (at the time of the data collection)
Downvotes - number of downvotes the user received (at the time of the data collection)


We employed the September 2011 dump of anonymized creative commons questions and answers from the Stack Exchange family of websites at http://stackexchange.com/sites.  We only extracted requests from post comments.

-----------------------------------

<===> stack-exchange.annotated.csv

Subset of the StackExchange requests for which we collected politeness annotations. The fields are:

Community - the StackExchange community in which the request was posted (e.g., "092011 Cooking" corresponds comments to posts from the the cooking StackExchange site http://cooking.stackexchange.com/)
Id - ID of the request, unique withing the community (can be used together with the Community field to match the requests with the metadata in stack-exchange.requests.csv)
Request - text of the request
Score[1-5] - politeness scores assigned by each of the 5 annotators, between 1 (very impolite) and 25 (very polite)
TurkId[1-5]	- the Amazon Mechanical Turk ids of each of the 5 annotators
Normalized Score - the normalized politeness score of the requests (after normalizing each annotator's score; details in Section 2 of the paper)


-----------------------------------

<===> stack-exchange.roles.csv

Author roles for the subset of requests originating in the Stack Overflow community. This information is used for defining the experimental settings described below.  The fields are:

Community - always "092011 Stack Overflow" in this file
Id - ID of the request, unique withing the community 
UserId - StackOverflow user ID
PostId - Stack Overflow ID of the post which is commented on
Author Role - Role of the author of the request: "Question-asker" (the author also posted the original question), "Comment-on-own-post user" (the author commented on her own post), "Other user".  

-----------------------------------



C) Experimental setting:

Here we describe the details of the experimental setting for various experiments presented in the above mentioned paper.

<> Stack Exchange question-asker experiment.
	1) Restrict analysis to Stack Overflow community.	
	2) Discard comments written by the author of the commented post (Author Role:"Comment-on-own-post").
	3) Divide requests into those posted by the "question askers" (requests posted by users who asked the original question; Author Role:"Question-asker") and other requests (Author Role:"Other user").

<> Stack Exchange reputation experiment.
	1) Restrict analysis to Stack Overflow community.
	2) Restrict analysis to requests posted within 6 months of the data dump (since we used a dump from September, 2011, we restricted analysis to requests posted after March, 2011).
	3) Discard comments written by the author of the commented post (Author Role:"Comment-on-own-post").
	4) Restrict analysis to requests posted by the "question askers" (requests posted by users who asked the original question; Author Role:"Question-asker").
	5) Define low-, middle-, and high-reputation by the distribution of users being considered. The top quartile of users by reputation are high-reputation users, the middle two quartiles are middle-reputation users, and the lowest quartile are low-reputation users.

<> Stack Exchange programming language experiment.
	1) Restrict analysis to Stack Overflow community.
	2) Use tags to determine which posts belong to which programming language sub-community.



D) Contact:

Please email any questions to: cristiand@cs.stanford.edu (Cristian Danescu-Niculescu-Mizil)



