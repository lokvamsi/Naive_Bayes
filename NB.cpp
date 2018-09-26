#include<stdio.h>
#include<fstream>
#include<iostream>
#include<vector>
#include<cstdio>
#include<string>
#include<math.h>
#include<algorithm>

using namespace std;

int class_1_wc[89527];	///Array containing word counts for each word in vocab, for class 1 i.e positive reviews
int class_2_wc[89527];	///Array containing word counts for each word in vocab, for class 2 i.e negative reviews
float class_1_log[89527];	///log of counts as defined above
float class_2_log[89527];

/***
 * Structure of word
 * index - vocabulary reference number
 * freq - number of times it occurs
 */
struct  word{
	int index=0;
	int freq=0;
};
/***
 * Structure of a review
 * w - vector of words
 * rating - rating given by reviewer
 */
struct review {
	vector<word> w;
	int rating=0;
};

/***
 * Take input from stop words file into vector containing indices of stop words
 * @param file File path
 * @param n number of stop words to take
 * @return vector containing stop word indices
 */
std::vector<int> Input_stopw(std::string &file,int n) {
	std::vector<int> stopw_v;
	ifstream inFile;
	inFile.open(file.c_str());

	char END_OF_LINE = '\n';

	char singleCharacter;
	int index;
	int j=0;

	while(singleCharacter != EOF && j<n) {

		inFile>>index;
		stopw_v.push_back(index);
		inFile.get(singleCharacter);
		j++;
	}
	return stopw_v;
}

/***
 * Take input for reviews formatted as per BoW model
 * @param file File path
 * @param n number of reviews to take
 * @return vector of 'review's representing all the data in file
 */
std::vector<review> Input(std::string &file,int n){

	std::vector<review> dataset;
	ifstream inFile;
	inFile.open(file.c_str());

	char END_OF_REVIEW = '\n';
	char singleCharacter;
	int rating;
	int i=0;
	int j=0;

	while(singleCharacter != EOF && j<n)	{
		struct review rev;
		inFile>>rating;
		rev.rating=rating;
		singleCharacter=' ';

		while (singleCharacter != END_OF_REVIEW) {

			struct word temp_word;
			int word_id;
			int word_freq;

			inFile>>word_id;
			inFile.get(singleCharacter);
			inFile>>word_freq;
			inFile.get(singleCharacter);

			(temp_word).index=word_id;
			(temp_word).freq=word_freq;

			rev.w.push_back(temp_word);

		}

		dataset.push_back(rev);
		j++;

	}

	return dataset;
}

/***
 * Print a review
 * @param rev review to be printed
 */
void printReview(struct review rev) {
	cout<<"Rating: "<<rev.rating<<"\n";
	for(int i=0;i<rev.w.size();i++) {
		cout<<rev.w[i].index<<":"<<rev.w[i].freq<<" ";
	}
	cout<<"\n";
}

/***
 * Fills word count array for both class 1 and 2 based on dataset
 * @param dataset training review data
 */
void countWc(vector<review> dataset) {

	for(int i=0;i<dataset.size();i++) {
		if(dataset[i].rating>=5)
			for(int j=0;j<dataset[i].w.size();j++) {
				class_1_wc[dataset[i].w[j].index]+=dataset[i].w[j].freq;
			}
		else
			for(int j=0;j<dataset[i].w.size();j++) {
				class_2_wc[dataset[i].w[j].index]+=dataset[i].w[j].freq;
			}
	}

}
/***
 * Fills word count as per Binary NB model
 * @param dataset training review data
 */
void countWc_B(vector<review> dataset) {

	for(int i=0;i<dataset.size();i++) {
		if(dataset[i].rating>=5)
			for(int j=0;j<dataset[i].w.size();j++) {
				if(dataset[i].w[j].freq > 0)
					class_1_wc[dataset[i].w[j].index]+=1;
			}
		else
			for(int j=0;j<dataset[i].w.size();j++) {
				if(dataset[i].w[j].freq > 0)
					class_2_wc[dataset[i].w[j].index]+=1;
			}
	}

}
/***
 * Fill word counts taking into consideration stop word elimination
 * @param dataset training data
 * @param stopw_v vector containing indices of stop words
 */
void countWc_stopw(vector<review> dataset,vector<int> stopw_v) {

for(int i=0;i<dataset.size();i++) {
	if(dataset[i].rating>=5)
		for(int j=0;j<dataset[i].w.size();j++) {
			if(std::find(stopw_v.begin(),stopw_v.end(),dataset[i].w[j].index) != stopw_v.end())
				class_1_wc[dataset[i].w[j].index]=0;
			else
				class_1_wc[dataset[i].w[j].index]+=dataset[i].w[j].freq;
		}
	else
		for(int j=0;j<dataset[i].w.size();j++) {
			if(std::find(stopw_v.begin(),stopw_v.end(),dataset[i].w[j].index) != stopw_v.end())
				class_2_wc[dataset[i].w[j].index]=0;
			else
				class_2_wc[dataset[i].w[j].index]+=dataset[i].w[j].freq;

		}
	}
}

/***
 * Fills log likelihood array of each class for each word
 * @param dataset training data
 */
void loglikelihood(std::vector<review> dataset){

	int c1wordcount=0;
	int c2wordcount=0;
	int vocabcount=89527;

	for(int i=0;i<89527;i++){
		c1wordcount+=class_1_wc[i];
	}
	for(int i=0;i<89527;i++){
		c2wordcount+=class_2_wc[i];
	}

	for(int i=0;i<89527;i++){
		float temp=(float)c1wordcount+(float)vocabcount;
		class_1_log[i]=(float)log10((float)(class_1_wc[i]+1)/temp);
	}
	for(int i=0;i<89527;i++){
		float temp=(float)c1wordcount+(float)vocabcount;
		class_2_log[i]=log10((float)(class_2_wc[i]+1)/temp);
	}

}

/***
 * Returns most probable class based on trained log likelihood function
 * @param test_doc test data
 * @param logp_1 log prior probability value for class 1
 * @param logp_2 log prior probability value for class 2
 * @return
 */
int testNB(struct review test_doc,float logp_1,float logp_2) {
	float sum_1=logp_1;
	float sum_2=logp_2;

	for(int j=0;j<test_doc.w.size();j++) {
		sum_1+=class_1_log[test_doc.w[j].index];
		sum_2+=class_2_log[test_doc.w[j].index];
	}

	if(sum_1>sum_2)
		return 0;
	else
		return 1;
}

/***
 * Function to print statistics for class 1 results on test data
 * @param test_set test dataset
 */
void acc_test_1(vector<review> test_set) {


	int c1_doc_num = 12500;
	int c2_doc_num = 12500;
	int tot_doc_num =c1_doc_num+c2_doc_num;

	float c1_log_prior=log10((float)c1_doc_num/tot_doc_num);
	float c2_log_prior=log10((float)c2_doc_num/tot_doc_num);

	int false_pos=0;
	int true_pos=0;
	int false_neg=0;
	int true_neg=0;

	for(int i=0;i<test_set.size();i++) {
		int c=testNB(test_set[i],c1_log_prior,c2_log_prior);

		if(c==0 && test_set[i].rating>=5)
			true_pos++;
		else if(c==0 && test_set[i].rating<5)
			false_pos++;
		else if(c==1 && test_set[i].rating<5)
			true_neg++;
		else if(c==1 && test_set[i].rating>=5)
			false_neg++;
	}

	float accuracy = (float) (true_pos + true_neg)/(true_pos + true_neg + false_pos + false_neg);
	float precision = (float) (true_pos)/ (true_pos + false_pos);
	float recall = (float) 	(true_pos)/ (true_pos + false_neg);
	cout<<"Accuracy is:"<< accuracy << "\n";
	cout<<"Precision is:"<< precision << "\n";
	cout<<"Recall is:"<< recall << "\n";
	cout<<"F measure is:"<< 2* precision*recall/(precision + recall)<<"\n";
}


/***
 * Prints statistics for class 2 on test data
 * @param test_set test dataset
 */
void acc_test_2(vector<review> test_set) {


	int c1_doc_num = 12500;
	int c2_doc_num = 12500;
	int tot_doc_num =c1_doc_num+c2_doc_num;

	float c1_log_prior=log10((float)c1_doc_num/tot_doc_num);
	float c2_log_prior=log10((float)c2_doc_num/tot_doc_num);

	int false_pos=0;
	int true_pos=0;
	int false_neg=0;
	int true_neg=0;

	for(int i=0;i<test_set.size();i++) {
		int c=testNB(test_set[i],c1_log_prior,c2_log_prior);

		if(c==1 && test_set[i].rating<5)
			true_pos++;
		else if(c==1 && test_set[i].rating>=5)
			false_pos++;
		else if(c==0 && test_set[i].rating>=5)
			true_neg++;
		else if(c==0 && test_set[i].rating<5)
			false_neg++;
	}

	float accuracy = (float) (true_pos + true_neg)/(true_pos + true_neg + false_pos + false_neg);
	float precision = (float) (true_pos)/ (true_pos + false_pos);
	float recall = (float) 	(true_pos)/ (true_pos + false_neg);
	cout<<"Accuracy is:"<< accuracy << "\n";
	cout<<"Precision is:"<< precision << "\n";
	cout<<"Recall is:"<< recall << "\n";
	cout<<"F measure is:"<< 2* precision*recall/(precision + recall)<<"\n";
}




/***
 * Main driver function, runs through each method printing stats on test data
 * @return
 */
int main(){

std::string filename_data="labeledBow.txt";
std::string filename_test="labeledBow.feat";
std::string filename_stopw="stopwordindex.txt";

std::vector<review> dataset=Input(filename_data,25000);
std::vector<review> test_set=Input(filename_test,25000);
std::vector<int> stopw_index=Input_stopw(filename_stopw,202);

cout<<"Naive bayes Statistics on test set:\n";

countWc(dataset);

loglikelihood(dataset);

cout<<"Class 1:"<<"\n";

acc_test_1(test_set);

cout<<"Class 2:"<<"\n";

acc_test_2(test_set);

cout<<"Binary modification statistics on test set: \n";

countWc_B(dataset);

loglikelihood(dataset);

cout<<"Class 1:"<<"\n";

acc_test_1(test_set);

cout<<"Class 2:"<<"\n";

acc_test_2(test_set);


cout<<"Stop word modification statistics on test set: \n";

countWc_stopw(dataset,stopw_index);


loglikelihood(dataset);

cout<<"Class 1:"<<"\n";

acc_test_1(test_set);

cout<<"Class 2:"<<"\n";

acc_test_2(test_set);


return 0;
}
