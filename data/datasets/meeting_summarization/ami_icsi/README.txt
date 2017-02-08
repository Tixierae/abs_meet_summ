Data for extractive summarization / sentence classification task on the
meeting domain. Contains subsets of the AMI and ICSI Meeting Corpora
depending on whether there is annotation available. 

The data format is tab separated values in that order: 
0	DA-id 
1	start-time
2	end-time
3	speaker-id
4	job/education
5	DA-type 
6	{-,+}1 indicating the importance
7	number of links of this da to abstractive summary
8	spoken {words,vocalizations}

The link-files (abstractive->DA) are structured as follows:
<#>\t<abstractive sentence>
<DA-id 1>
<DA-id 2>
...
<DA-id #>
where # is the number of DAs linked to the sentence.

Examples:
[koried@date data]$ head -n4 ami/ES2002a.da
ES2002a.B.dialog-act.dharshi.1  50.42   50.99   B       PM      stl     -1      0               Okay
ES2002a.B.dialog-act.dharshi.2  53.56   53.96   B       PM      stl     -1      0               Right
ES2002a.B.dialog-act.dharshi.3  55.415  60.35   B       PM      inf     +1      1               {vocalsound} Um well this is the kick-off meeting for our our project .
ES2002a.B.dialog-act.dharshi.16 60.35   64.16   B       PM      stl     -1      0               Um {vocalsound} and um

Dialog acts for AMI Meeting Corpus:
bck Backchannel
stl Stall
fra Fragment
inf Inform
el.inf Elicit-Inform
sug Suggest
off Offer
el.sug Elicit-Offer-Or-Suggestion
ass Assess
el.ass Elicit-Assessment
und Comment-About-Understanding
el.und Elicit-Comment-Understanding
be.pos Be-Positive
be.neg Be-Negative
oth Other

Job descriptions for AMI:
UI User Interface
ID Industrial Design
PM Program Manager
ME Marketing Expert

[koried@date data]$ head -n4 icsi/Bdb001.da
Bdb001.C.dialogueact0   0.216   5.914   C       Grad    z z       -1      0       Yeah , we had a long discussion about how much w how easy we want to make it for people to bleep things out .
Bdb001.C.dialogueact1   5.914   6.254   C       Grad    z z       -1      0       So {disfmarker}
Bdb001.C.dialogueact2   8.339   9.499   C       Grad    z z       -1      0       Morgan wants to make it hard .
Bdb001.D.dialogueact3   13.38   14.41   D       PhD     z z       -1      0       It {disfmarker} it doesn't {disfmarker}

Dialog acts for ICSI Meeting Corpus
http://www.icsi.berkeley.edu/cgi-bin/pubs/publication.pl?ID=000231

Directory contents:
ami/				AMI Meeting Corpus
icsi/				ICSI Meeting Corpus
	*.da			dialog acts as described above
	*.adj			dialog adjacencies with adj-type
	*.{#}			Links between abstractive summaries and DA
					#-ami:  abstract, problems, decisions, actions
					#-icsi: abstract, problems, decisions, progress
AMIExporter.jar		Java program to extract AMI data
ICSIExporter.jar	Java program to extract ICSI data
ami_export.sh		ready-to-invoke script to extract AMI meeting
icsi_export.sh		ready-to-invoke script to extract ICSI meeting
list.ami			list of available AMI meetings
list.icsi			list of available ICSI meetings
example.pl			example Perl file how to parse and use the dialog acts

Subdir contents generated with
for i in `cat list.ami`; do ./ami_export.sh $i > ami/$i.da; done
for i in `cat list.icsi`; do ./icsi_export.sh $i > icsi/$i.da; done
for i in `cat list.ami`; do ./ami_adjacencies.pl $i > ami/$i.adj; done

for i in `cat list.ami`; do 
   for j in "abstract" "problems" "decisions" "actions"; do
     ./ami_abstr_extr.pl $i $j > ami/$i.$j
   done
done

for i in `cat list.icsi`; do 
   for j in "abstract" "problems" "decisions" "progress"; do
     ./icsi_abstr_extr.pl $i $j > icsi/$i.$j
   done
done

for i in `cat lists/list.icsi.test`; do 
	for j in "beata" "s9553330" "vkaraisk"; do 
		for k in "abstract" "problems" "decisions" "progress"; do
			echo "$i $j $k"
			./icsi_abstr_extr_spk.pl $i $j $k > icsi/$i.$j.$k
		done
	done
done

for i in `cat list.icsi.test`; do
	for j in "beata" "s9553330" "vkaraisk"; do
		for k in "abstract" "problems" "decisions" "progress"; do
			./icsi_abstr_ducref_spk.pl $j $i $k > icsi/$i.ducref.$j.$k
		done
	done
done

for i in `cat lists/list.icsi.test`; do 
	for j in "beata" "s9553330" "vkaraisk"; do 
		./icsi_extractive2.pl $j $i > icsi/$i.ducref.summlink.$j
	done 
done

for i in `cat lists/list.icsi.test`; do 
	for j in "beata" "s9553330" "vkaraisk"; do 
		./icsi_extractive1.pl $j $i > icsi/$i.ducref.extsumm.$j
	done
done

# RTTMX to DA files
/n/abbott/dl/drspeech/GALE/bu/AA/English/mrda/stt_auto/
for i in `cat lists/list.icsi`; do
	scripts/generate_asr_icsi.pl icsi/$i.da /n/abbott/dl/drspeech/GALE/bu/AA/English/mrda/stt_auto/$i.rttmx.gz > icsi/$i.da-asr
done

for i in `cat lists/list.ami`; do
	scripts/generate_asr_ami.pl ami/$i.da > ami/$i.da-asr
done

