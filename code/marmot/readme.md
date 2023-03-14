To get annotation from an existing model:
+ `` java -cp marmot.jar marmot.morph.cmd.Annotator --model-file pl.marmot --test-file form-index=0,text.txt --pred-file text.out.txt``

However, the annotation does not fit. I need to train my own model.
Preproc the data:
+ ``cat pl_pdb-ud-train.conllu | grep -v "^#" | cut -f 2,4,5 | tr '\t' ' ' > train.txt``
+ ``cat pl_pdb-ud-test.conllu | grep -v "^#" | cut -f 2,4,5 | tr '\t' ' ' > test.txt`` move to the Marmot folder
+ ``python3 ../preproc_bert.py train.txt`` from the Marmot folder to remove ranges
+ ``python3 ../preproc_bert.py test.txt``

Train the model:
+ ``java -Xmx5G -cp marmot.jar marmot.morph.cmd.Trainer -train-file form-index=0,tag-index=1,morph-index=2,train.txt -tag-morph true -model-file plTRAINED.marmot`` make sure that the indices fit and choose an output name

Test the model:
+ ``java -cp marmot.jar marmot.morph.cmd.Annotator --model-file pl_custom.marmot --test-file form-index=0,test.txt --pred-file test.out.txt``

Preproc the historical test data:
+ 
