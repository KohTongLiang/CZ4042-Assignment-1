Ӫ
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
o

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name	1653682*
value_dtype0	
?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_nametable_1653605*
value_dtype0	
q
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	1653794*
value_dtype0	
?
MutableHashTable_1MutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_1653717*
value_dtype0	
q
hash_table_2HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	1653906*
value_dtype0	
?
MutableHashTable_2MutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_1653829*
value_dtype0	
?
embedding_236/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameembedding_236/embeddings
?
,embedding_236/embeddings/Read/ReadVariableOpReadVariableOpembedding_236/embeddings*
_output_shapes

:*
dtype0
?
embedding_237/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameembedding_237/embeddings
?
,embedding_237/embeddings/Read/ReadVariableOpReadVariableOpembedding_237/embeddings*
_output_shapes

:*
dtype0
?
embedding_238/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameembedding_238/embeddings
?
,embedding_238/embeddings/Read/ReadVariableOpReadVariableOpembedding_238/embeddings*
_output_shapes

:*
dtype0
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
d
mean_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean_1
]
mean_1/Read/ReadVariableOpReadVariableOpmean_1*
_output_shapes
:*
dtype0
l

variance_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance_1
e
variance_1/Read/ReadVariableOpReadVariableOp
variance_1*
_output_shapes
:*
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0	
d
mean_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean_2
]
mean_2/Read/ReadVariableOpReadVariableOpmean_2*
_output_shapes
:*
dtype0
l

variance_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance_2
e
variance_2/Read/ReadVariableOpReadVariableOp
variance_2*
_output_shapes
:*
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0	
d
mean_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean_3
]
mean_3/Read/ReadVariableOpReadVariableOpmean_3*
_output_shapes
:*
dtype0
l

variance_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance_3
e
variance_3/Read/ReadVariableOpReadVariableOp
variance_3*
_output_shapes
:*
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0	
d
mean_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean_4
]
mean_4/Read/ReadVariableOpReadVariableOpmean_4*
_output_shapes
:*
dtype0
l

variance_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance_4
e
variance_4/Read/ReadVariableOpReadVariableOp
variance_4*
_output_shapes
:*
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0	
d
mean_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean_5
]
mean_5/Read/ReadVariableOpReadVariableOpmean_5*
_output_shapes
:*
dtype0
l

variance_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance_5
e
variance_5/Read/ReadVariableOpReadVariableOp
variance_5*
_output_shapes
:*
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0	
z
dense_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:)
* 
shared_namedense_59/kernel
s
#dense_59/kernel/Read/ReadVariableOpReadVariableOpdense_59/kernel*
_output_shapes

:)
*
dtype0
r
dense_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_59/bias
k
!dense_59/bias/Read/ReadVariableOpReadVariableOpdense_59/bias*
_output_shapes
:
*
dtype0
z
dense_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_60/kernel
s
#dense_60/kernel/Read/ReadVariableOpReadVariableOpdense_60/kernel*
_output_shapes

:
*
dtype0
r
dense_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_60/bias
k
!dense_60/bias/Read/ReadVariableOpReadVariableOpdense_60/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_9
[
count_9/Read/ReadVariableOpReadVariableOpcount_9*
_output_shapes
: *
dtype0
?
Adam/embedding_236/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/embedding_236/embeddings/m
?
3Adam/embedding_236/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_236/embeddings/m*
_output_shapes

:*
dtype0
?
Adam/embedding_237/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/embedding_237/embeddings/m
?
3Adam/embedding_237/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_237/embeddings/m*
_output_shapes

:*
dtype0
?
Adam/embedding_238/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/embedding_238/embeddings/m
?
3Adam/embedding_238/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_238/embeddings/m*
_output_shapes

:*
dtype0
?
Adam/dense_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:)
*'
shared_nameAdam/dense_59/kernel/m
?
*Adam/dense_59/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/m*
_output_shapes

:)
*
dtype0
?
Adam/dense_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_59/bias/m
y
(Adam/dense_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_60/kernel/m
?
*Adam/dense_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_60/bias/m
y
(Adam/dense_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_236/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/embedding_236/embeddings/v
?
3Adam/embedding_236/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_236/embeddings/v*
_output_shapes

:*
dtype0
?
Adam/embedding_237/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/embedding_237/embeddings/v
?
3Adam/embedding_237/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_237/embeddings/v*
_output_shapes

:*
dtype0
?
Adam/embedding_238/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/embedding_238/embeddings/v
?
3Adam/embedding_238/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_238/embeddings/v*
_output_shapes

:*
dtype0
?
Adam/dense_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:)
*'
shared_nameAdam/dense_59/kernel/v
?
*Adam/dense_59/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/v*
_output_shapes

:)
*
dtype0
?
Adam/dense_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_59/bias/v
y
(Adam/dense_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_60/kernel/v
?
*Adam/dense_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_60/bias/v
y
(Adam/dense_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
\
Const_3Const*
_output_shapes

:*
dtype0*
valueB*ur?B
\
Const_4Const*
_output_shapes

:*
dtype0*
valueB*3rD
\
Const_5Const*
_output_shapes

:*
dtype0*
valueB*?P?
\
Const_6Const*
_output_shapes

:*
dtype0*
valueB*d^>
\
Const_7Const*
_output_shapes

:*
dtype0*
valueB*&?3A
\
Const_8Const*
_output_shapes

:*
dtype0*
valueB*wz?A
\
Const_9Const*
_output_shapes

:*
dtype0*
valueB*_?<
]
Const_10Const*
_output_shapes

:*
dtype0*
valueB*y??7
]
Const_11Const*
_output_shapes

:*
dtype0*
valueB*:r?;
]
Const_12Const*
_output_shapes

:*
dtype0*
valueB*???9
]
Const_13Const*
_output_shapes

:*
dtype0*
valueB*z?B
]
Const_14Const*
_output_shapes

:*
dtype0*
valueB*M?'C
J
Const_15Const*
_output_shapes
: *
dtype0	*
value	B	 R 
J
Const_16Const*
_output_shapes
: *
dtype0	*
value	B	 R 
J
Const_17Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Const_18Const*
_output_shapes
:*
dtype0	*u
valuelBj	"`       
                            	                                                 
?
Const_19Const*
_output_shapes
:*
dtype0	*u
valuelBj	"`                                                        	       
                     
?
Const_20Const*
_output_shapes
:+*
dtype0*?
value?B?+B4 ROOM, Model AB5 ROOM, ImprovedB3 ROOM, New GenerationB3 ROOM, ImprovedB3 ROOM, Model AB4 ROOM, Premium ApartmentB4 ROOM, New GenerationBEXECUTIVE, ApartmentB5 ROOM, Premium ApartmentB4 ROOM, SimplifiedBEXECUTIVE, MaisonetteB3 ROOM, StandardB5 ROOM, Model AB4 ROOM, Model A2B4 ROOM, ImprovedB3 ROOM, SimplifiedB5 ROOM, StandardBEXECUTIVE, Premium ApartmentB2 ROOM, Model AB5 ROOM, DBSSB4 ROOM, DBSSB3 ROOM, Premium ApartmentB2 ROOM, StandardB2 ROOM, ImprovedB3 ROOM, DBSSB5 ROOM, Model A-MaisonetteB4 ROOM, Type S1B5 ROOM, Adjoined flatB5 ROOM, Type S2BEXECUTIVE, Adjoined flatB3 ROOM, TerraceB"MULTI-GENERATION, Multi GenerationB1 ROOM, ImprovedB4 ROOM, StandardB4 ROOM, Premium Apartment LoftB2 ROOM, Premium ApartmentB5 ROOM, Improved-MaisonetteB4 ROOM, Adjoined flatBEXECUTIVE, Premium MaisonetteB5 ROOM, Premium Apartment LoftB2 ROOM, 2-roomB4 ROOM, TerraceB2 ROOM, DBSS
?
Const_21Const*
_output_shapes
:+*
dtype0	*?
value?B?	+"?                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       
?
Const_22Const*
_output_shapes
:*
dtype0*?
value?B?B04 TO 06B07 TO 09B10 TO 12B01 TO 03B13 TO 15B16 TO 18B19 TO 21B22 TO 24B25 TO 27B28 TO 30B31 TO 33B37 TO 39B34 TO 36B40 TO 42B46 TO 48B43 TO 45B49 TO 51
?
Const_23Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                        	       
                                                        
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_18Const_19*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference_<lambda>_2043762
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference_<lambda>_2043767
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_1Const_20Const_21*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference_<lambda>_2043775
?
PartitionedCall_1PartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference_<lambda>_2043780
?
StatefulPartitionedCall_2StatefulPartitionedCallhash_table_2Const_22Const_23*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference_<lambda>_2043788
?
PartitionedCall_2PartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference_<lambda>_2043793
?
NoOpNoOp^PartitionedCall^PartitionedCall_1^PartitionedCall_2^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0	*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?
AMutableHashTable_1_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable_1*
Tkeys0*
Tvalues0	*%
_class
loc:@MutableHashTable_1*
_output_shapes

::
?
AMutableHashTable_2_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable_2*
Tkeys0*
Tvalues0	*%
_class
loc:@MutableHashTable_2*
_output_shapes

::
?\
Const_24Const"/device:CPU:0*
_output_shapes
: *
dtype0*?[
value?[B?[ B?[
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer_with_weights-6
layer-18
layer_with_weights-7
layer-19
layer_with_weights-8
layer-20
layer_with_weights-9
layer-21
layer_with_weights-10
layer-22
layer_with_weights-11
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
	optimizer
regularization_losses
	variables
trainable_variables
 	keras_api
!
signatures
 
 
 
3
"lookup_table
#token_counts
$	keras_api
3
%lookup_table
&token_counts
'	keras_api
3
(lookup_table
)token_counts
*	keras_api
b
+
embeddings
,regularization_losses
-	variables
.trainable_variables
/	keras_api
b
0
embeddings
1regularization_losses
2	variables
3trainable_variables
4	keras_api
b
5
embeddings
6regularization_losses
7	variables
8trainable_variables
9	keras_api
 
 
 
 
 
 
R
:regularization_losses
;	variables
<trainable_variables
=	keras_api
R
>regularization_losses
?	variables
@trainable_variables
A	keras_api
R
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
?
F
_keep_axis
G_reduce_axis
H_reduce_axis_mask
I_broadcast_shape
Jmean
J
adapt_mean
Kvariance
Kadapt_variance
	Lcount
M	keras_api
?
N
_keep_axis
O_reduce_axis
P_reduce_axis_mask
Q_broadcast_shape
Rmean
R
adapt_mean
Svariance
Sadapt_variance
	Tcount
U	keras_api
?
V
_keep_axis
W_reduce_axis
X_reduce_axis_mask
Y_broadcast_shape
Zmean
Z
adapt_mean
[variance
[adapt_variance
	\count
]	keras_api
?
^
_keep_axis
__reduce_axis
`_reduce_axis_mask
a_broadcast_shape
bmean
b
adapt_mean
cvariance
cadapt_variance
	dcount
e	keras_api
?
f
_keep_axis
g_reduce_axis
h_reduce_axis_mask
i_broadcast_shape
jmean
j
adapt_mean
kvariance
kadapt_variance
	lcount
m	keras_api
?
n
_keep_axis
o_reduce_axis
p_reduce_axis_mask
q_broadcast_shape
rmean
r
adapt_mean
svariance
sadapt_variance
	tcount
u	keras_api
R
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
h

zkernel
{bias
|regularization_losses
}	variables
~trainable_variables
	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate+m?0m?5m?zm?{m?	?m?	?m?+v?0v?5v?zv?{v?	?v?	?v?
 
?
+3
04
55
J6
K7
L8
R9
S10
T11
Z12
[13
\14
b15
c16
d17
j18
k19
l20
r21
s22
t23
z24
{25
?26
?27
3
+0
01
52
z3
{4
?5
?6
?
?layer_metrics
?layers
regularization_losses
?non_trainable_variables
?metrics
	variables
 ?layer_regularization_losses
trainable_variables
 

?_initializer
><
table3layer_with_weights-0/token_counts/.ATTRIBUTES/table
 

?_initializer
><
table3layer_with_weights-1/token_counts/.ATTRIBUTES/table
 

?_initializer
><
table3layer_with_weights-2/token_counts/.ATTRIBUTES/table
 
hf
VARIABLE_VALUEembedding_236/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

+0

+0
?
?layer_metrics
?layers
,regularization_losses
?non_trainable_variables
?metrics
-	variables
 ?layer_regularization_losses
.trainable_variables
hf
VARIABLE_VALUEembedding_237/embeddings:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

00

00
?
?layer_metrics
?layers
1regularization_losses
?non_trainable_variables
?metrics
2	variables
 ?layer_regularization_losses
3trainable_variables
hf
VARIABLE_VALUEembedding_238/embeddings:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

50

50
?
?layer_metrics
?layers
6regularization_losses
?non_trainable_variables
?metrics
7	variables
 ?layer_regularization_losses
8trainable_variables
 
 
 
?
?layer_metrics
?layers
:regularization_losses
?non_trainable_variables
?metrics
;	variables
 ?layer_regularization_losses
<trainable_variables
 
 
 
?
?layer_metrics
?layers
>regularization_losses
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
@trainable_variables
 
 
 
?
?layer_metrics
?layers
Bregularization_losses
?non_trainable_variables
?metrics
C	variables
 ?layer_regularization_losses
Dtrainable_variables
 
 
 
 
NL
VARIABLE_VALUEmean4layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvariance8layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcount5layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_14layer_with_weights-7/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_18layer_with_weights-7/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_15layer_with_weights-7/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_24layer_with_weights-8/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_28layer_with_weights-8/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_25layer_with_weights-8/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_34layer_with_weights-9/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_38layer_with_weights-9/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_35layer_with_weights-9/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
QO
VARIABLE_VALUEmean_45layer_with_weights-10/mean/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE
variance_49layer_with_weights-10/variance/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_46layer_with_weights-10/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
QO
VARIABLE_VALUEmean_55layer_with_weights-11/mean/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE
variance_59layer_with_weights-11/variance/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_56layer_with_weights-11/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
?
?layer_metrics
?layers
vregularization_losses
?non_trainable_variables
?metrics
w	variables
 ?layer_regularization_losses
xtrainable_variables
\Z
VARIABLE_VALUEdense_59/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_59/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

z0
{1

z0
{1
?
?layer_metrics
?layers
|regularization_losses
?non_trainable_variables
?metrics
}	variables
 ?layer_regularization_losses
~trainable_variables
\Z
VARIABLE_VALUEdense_60/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_60/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
?
J3
K4
L5
R6
S7
T8
Z9
[10
\11
b12
c13
d14
j15
k16
l17
r18
s19
t20
 
?0
?1
?2
?3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_74keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_84keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_94keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUEAdam/embedding_236/embeddings/mVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_237/embeddings/mVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_238/embeddings/mVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_59/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_59/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_60/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_60/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_236/embeddings/vVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_237/embeddings/vVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_238/embeddings/vVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_59/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_59/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_60/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_60/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
!serving_default_degree_centralityPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_dist_to_dhobyPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
#serving_default_dist_to_nearest_stnPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
&serving_default_eigenvector_centralityPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_flat_model_typePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_floor_area_sqmPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
x
serving_default_monthPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
%serving_default_remaining_lease_yearsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_storey_rangePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_3StatefulPartitionedCall!serving_default_degree_centralityserving_default_dist_to_dhoby#serving_default_dist_to_nearest_stn&serving_default_eigenvector_centralityserving_default_flat_model_typeserving_default_floor_area_sqmserving_default_month%serving_default_remaining_lease_yearsserving_default_storey_rangehash_table_2Consthash_table_1Const_1
hash_tableConst_2embedding_238/embeddingsembedding_237/embeddingsembedding_236/embeddingsConst_3Const_4Const_5Const_6Const_7Const_8Const_9Const_10Const_11Const_12Const_13Const_14dense_59/kerneldense_59/biasdense_60/kerneldense_60/bias*-
Tin&
$2"				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	 !*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_2043097
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filename?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1AMutableHashTable_1_lookup_table_export_values/LookupTableExportV2CMutableHashTable_1_lookup_table_export_values/LookupTableExportV2:1AMutableHashTable_2_lookup_table_export_values/LookupTableExportV2CMutableHashTable_2_lookup_table_export_values/LookupTableExportV2:1,embedding_236/embeddings/Read/ReadVariableOp,embedding_237/embeddings/Read/ReadVariableOp,embedding_238/embeddings/Read/ReadVariableOpmean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOpmean_1/Read/ReadVariableOpvariance_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpmean_2/Read/ReadVariableOpvariance_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpmean_3/Read/ReadVariableOpvariance_3/Read/ReadVariableOpcount_3/Read/ReadVariableOpmean_4/Read/ReadVariableOpvariance_4/Read/ReadVariableOpcount_4/Read/ReadVariableOpmean_5/Read/ReadVariableOpvariance_5/Read/ReadVariableOpcount_5/Read/ReadVariableOp#dense_59/kernel/Read/ReadVariableOp!dense_59/bias/Read/ReadVariableOp#dense_60/kernel/Read/ReadVariableOp!dense_60/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount_6/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_8/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_9/Read/ReadVariableOp3Adam/embedding_236/embeddings/m/Read/ReadVariableOp3Adam/embedding_237/embeddings/m/Read/ReadVariableOp3Adam/embedding_238/embeddings/m/Read/ReadVariableOp*Adam/dense_59/kernel/m/Read/ReadVariableOp(Adam/dense_59/bias/m/Read/ReadVariableOp*Adam/dense_60/kernel/m/Read/ReadVariableOp(Adam/dense_60/bias/m/Read/ReadVariableOp3Adam/embedding_236/embeddings/v/Read/ReadVariableOp3Adam/embedding_237/embeddings/v/Read/ReadVariableOp3Adam/embedding_238/embeddings/v/Read/ReadVariableOp*Adam/dense_59/kernel/v/Read/ReadVariableOp(Adam/dense_59/bias/v/Read/ReadVariableOp*Adam/dense_60/kernel/v/Read/ReadVariableOp(Adam/dense_60/bias/v/Read/ReadVariableOpConst_24*G
Tin@
>2<											*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_2044028
?	
StatefulPartitionedCall_5StatefulPartitionedCallsaver_filenameMutableHashTableMutableHashTable_1MutableHashTable_2embedding_236/embeddingsembedding_237/embeddingsembedding_238/embeddingsmeanvariancecountmean_1
variance_1count_1mean_2
variance_2count_2mean_3
variance_3count_3mean_4
variance_4count_4mean_5
variance_5count_5dense_59/kerneldense_59/biasdense_60/kerneldense_60/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount_6total_1count_7total_2count_8total_3count_9Adam/embedding_236/embeddings/mAdam/embedding_237/embeddings/mAdam/embedding_238/embeddings/mAdam/dense_59/kernel/mAdam/dense_59/bias/mAdam/dense_60/kernel/mAdam/dense_60/bias/mAdam/embedding_236/embeddings/vAdam/embedding_237/embeddings/vAdam/embedding_238/embeddings/vAdam/dense_59/kernel/vAdam/dense_59/bias/vAdam/dense_60/kernel/vAdam/dense_60/bias/v*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_2044203??
?
?
*__inference_model_29_layer_call_fn_2043427
inputs_0	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20:)


unknown_21:


unknown_22:


unknown_23:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*-
Tin&
$2"				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	 !*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_29_layer_call_and_return_conditional_losses_20427322
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : ::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
__inference_<lambda>_2043788:
6key_value_init1653905_lookuptableimportv2_table_handle2
.key_value_init1653905_lookuptableimportv2_keys4
0key_value_init1653905_lookuptableimportv2_values	
identity??)key_value_init1653905/LookupTableImportV2?
)key_value_init1653905/LookupTableImportV2LookupTableImportV26key_value_init1653905_lookuptableimportv2_table_handle.key_value_init1653905_lookuptableimportv2_keys0key_value_init1653905_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2+
)key_value_init1653905/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityz
NoOpNoOp*^key_value_init1653905/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init1653905/LookupTableImportV2)key_value_init1653905/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
,
__inference_<lambda>_2043793
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
J__inference_embedding_238_layer_call_and_return_conditional_losses_2042295

inputs	*
embedding_lookup_2042289:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2042289inputs",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/2042289*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@embedding_lookup/2042289*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
__inference_adapt_step_1661643
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNexts
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
??
?
E__inference_model_29_layer_call_and_return_conditional_losses_2043301
inputs_0	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8E
Astring_lookup_85_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_85_hash_table_lookup_lookuptablefindv2_default_value	E
Astring_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_84_hash_table_lookup_lookuptablefindv2_default_value	G
Cinteger_lookup_130_hash_table_lookup_lookuptablefindv2_table_handleH
Dinteger_lookup_130_hash_table_lookup_lookuptablefindv2_default_value	8
&embedding_238_embedding_lookup_2043222:8
&embedding_237_embedding_lookup_2043227:8
&embedding_236_embedding_lookup_2043232:
normalization_348_sub_y
normalization_348_sqrt_x
normalization_349_sub_y
normalization_349_sqrt_x
normalization_350_sub_y
normalization_350_sqrt_x
normalization_351_sub_y
normalization_351_sqrt_x
normalization_352_sub_y
normalization_352_sqrt_x
normalization_353_sub_y
normalization_353_sqrt_x9
'dense_59_matmul_readvariableop_resource:)
6
(dense_59_biasadd_readvariableop_resource:
9
'dense_60_matmul_readvariableop_resource:
6
(dense_60_biasadd_readvariableop_resource:
identity??dense_59/BiasAdd/ReadVariableOp?dense_59/MatMul/ReadVariableOp?dense_60/BiasAdd/ReadVariableOp?dense_60/MatMul/ReadVariableOp?embedding_236/embedding_lookup?embedding_237/embedding_lookup?embedding_238/embedding_lookup?6integer_lookup_130/hash_table_Lookup/LookupTableFindV2?4string_lookup_84/hash_table_Lookup/LookupTableFindV2?4string_lookup_85/hash_table_Lookup/LookupTableFindV2?
4string_lookup_85/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_85_hash_table_lookup_lookuptablefindv2_table_handleinputs_2Bstring_lookup_85_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_85/hash_table_Lookup/LookupTableFindV2?
string_lookup_85/IdentityIdentity=string_lookup_85/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_85/Identity?
4string_lookup_84/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleinputs_1Bstring_lookup_84_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_84/hash_table_Lookup/LookupTableFindV2?
string_lookup_84/IdentityIdentity=string_lookup_84/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_84/Identity?
6integer_lookup_130/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Cinteger_lookup_130_hash_table_lookup_lookuptablefindv2_table_handleinputs_0Dinteger_lookup_130_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????28
6integer_lookup_130/hash_table_Lookup/LookupTableFindV2?
integer_lookup_130/IdentityIdentity?integer_lookup_130/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup_130/Identity?
embedding_238/embedding_lookupResourceGather&embedding_238_embedding_lookup_2043222"string_lookup_85/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*9
_class/
-+loc:@embedding_238/embedding_lookup/2043222*+
_output_shapes
:?????????*
dtype02 
embedding_238/embedding_lookup?
'embedding_238/embedding_lookup/IdentityIdentity'embedding_238/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*9
_class/
-+loc:@embedding_238/embedding_lookup/2043222*+
_output_shapes
:?????????2)
'embedding_238/embedding_lookup/Identity?
)embedding_238/embedding_lookup/Identity_1Identity0embedding_238/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2+
)embedding_238/embedding_lookup/Identity_1?
embedding_237/embedding_lookupResourceGather&embedding_237_embedding_lookup_2043227"string_lookup_84/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*9
_class/
-+loc:@embedding_237/embedding_lookup/2043227*+
_output_shapes
:?????????*
dtype02 
embedding_237/embedding_lookup?
'embedding_237/embedding_lookup/IdentityIdentity'embedding_237/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*9
_class/
-+loc:@embedding_237/embedding_lookup/2043227*+
_output_shapes
:?????????2)
'embedding_237/embedding_lookup/Identity?
)embedding_237/embedding_lookup/Identity_1Identity0embedding_237/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2+
)embedding_237/embedding_lookup/Identity_1?
embedding_236/embedding_lookupResourceGather&embedding_236_embedding_lookup_2043232$integer_lookup_130/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*9
_class/
-+loc:@embedding_236/embedding_lookup/2043232*+
_output_shapes
:?????????*
dtype02 
embedding_236/embedding_lookup?
'embedding_236/embedding_lookup/IdentityIdentity'embedding_236/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*9
_class/
-+loc:@embedding_236/embedding_lookup/2043232*+
_output_shapes
:?????????2)
'embedding_236/embedding_lookup/Identity?
)embedding_236/embedding_lookup/Identity_1Identity0embedding_236/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2+
)embedding_236/embedding_lookup/Identity_1u
flatten_45/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_45/Const?
flatten_45/ReshapeReshape2embedding_236/embedding_lookup/Identity_1:output:0flatten_45/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_45/Reshapeu
flatten_46/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_46/Const?
flatten_46/ReshapeReshape2embedding_237/embedding_lookup/Identity_1:output:0flatten_46/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_46/Reshapeu
flatten_47/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_47/Const?
flatten_47/ReshapeReshape2embedding_238/embedding_lookup/Identity_1:output:0flatten_47/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_47/Reshape?
normalization_348/subSubinputs_3normalization_348_sub_y*
T0*'
_output_shapes
:?????????2
normalization_348/sub{
normalization_348/SqrtSqrtnormalization_348_sqrt_x*
T0*
_output_shapes

:2
normalization_348/Sqrt
normalization_348/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_348/Maximum/y?
normalization_348/MaximumMaximumnormalization_348/Sqrt:y:0$normalization_348/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_348/Maximum?
normalization_348/truedivRealDivnormalization_348/sub:z:0normalization_348/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_348/truediv?
normalization_349/subSubinputs_4normalization_349_sub_y*
T0*'
_output_shapes
:?????????2
normalization_349/sub{
normalization_349/SqrtSqrtnormalization_349_sqrt_x*
T0*
_output_shapes

:2
normalization_349/Sqrt
normalization_349/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_349/Maximum/y?
normalization_349/MaximumMaximumnormalization_349/Sqrt:y:0$normalization_349/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_349/Maximum?
normalization_349/truedivRealDivnormalization_349/sub:z:0normalization_349/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_349/truediv?
normalization_350/subSubinputs_5normalization_350_sub_y*
T0*'
_output_shapes
:?????????2
normalization_350/sub{
normalization_350/SqrtSqrtnormalization_350_sqrt_x*
T0*
_output_shapes

:2
normalization_350/Sqrt
normalization_350/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_350/Maximum/y?
normalization_350/MaximumMaximumnormalization_350/Sqrt:y:0$normalization_350/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_350/Maximum?
normalization_350/truedivRealDivnormalization_350/sub:z:0normalization_350/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_350/truediv?
normalization_351/subSubinputs_6normalization_351_sub_y*
T0*'
_output_shapes
:?????????2
normalization_351/sub{
normalization_351/SqrtSqrtnormalization_351_sqrt_x*
T0*
_output_shapes

:2
normalization_351/Sqrt
normalization_351/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_351/Maximum/y?
normalization_351/MaximumMaximumnormalization_351/Sqrt:y:0$normalization_351/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_351/Maximum?
normalization_351/truedivRealDivnormalization_351/sub:z:0normalization_351/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_351/truediv?
normalization_352/subSubinputs_7normalization_352_sub_y*
T0*'
_output_shapes
:?????????2
normalization_352/sub{
normalization_352/SqrtSqrtnormalization_352_sqrt_x*
T0*
_output_shapes

:2
normalization_352/Sqrt
normalization_352/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_352/Maximum/y?
normalization_352/MaximumMaximumnormalization_352/Sqrt:y:0$normalization_352/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_352/Maximum?
normalization_352/truedivRealDivnormalization_352/sub:z:0normalization_352/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_352/truediv?
normalization_353/subSubinputs_8normalization_353_sub_y*
T0*'
_output_shapes
:?????????2
normalization_353/sub{
normalization_353/SqrtSqrtnormalization_353_sqrt_x*
T0*
_output_shapes

:2
normalization_353/Sqrt
normalization_353/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_353/Maximum/y?
normalization_353/MaximumMaximumnormalization_353/Sqrt:y:0$normalization_353/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_353/Maximum?
normalization_353/truedivRealDivnormalization_353/sub:z:0normalization_353/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_353/truedivz
concatenate_58/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_58/concat/axis?
concatenate_58/concatConcatV2flatten_45/Reshape:output:0flatten_46/Reshape:output:0flatten_47/Reshape:output:0normalization_348/truediv:z:0normalization_349/truediv:z:0normalization_350/truediv:z:0normalization_351/truediv:z:0normalization_352/truediv:z:0normalization_353/truediv:z:0#concatenate_58/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????)2
concatenate_58/concat?
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:)
*
dtype02 
dense_59/MatMul/ReadVariableOp?
dense_59/MatMulMatMulconcatenate_58/concat:output:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_59/MatMul?
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_59/BiasAdd/ReadVariableOp?
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_59/BiasAdds
dense_59/ReluReludense_59/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_59/Relu?
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_60/MatMul/ReadVariableOp?
dense_60/MatMulMatMuldense_59/Relu:activations:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_60/MatMul?
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_60/BiasAdd/ReadVariableOp?
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_60/BiasAddt
IdentityIdentitydense_60/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp^embedding_236/embedding_lookup^embedding_237/embedding_lookup^embedding_238/embedding_lookup7^integer_lookup_130/hash_table_Lookup/LookupTableFindV25^string_lookup_84/hash_table_Lookup/LookupTableFindV25^string_lookup_85/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : ::::::::::::: : : : 2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2@
embedding_236/embedding_lookupembedding_236/embedding_lookup2@
embedding_237/embedding_lookupembedding_237/embedding_lookup2@
embedding_238/embedding_lookupembedding_238/embedding_lookup2p
6integer_lookup_130/hash_table_Lookup/LookupTableFindV26integer_lookup_130/hash_table_Lookup/LookupTableFindV22l
4string_lookup_84/hash_table_Lookup/LookupTableFindV24string_lookup_84/hash_table_Lookup/LookupTableFindV22l
4string_lookup_85/hash_table_Lookup/LookupTableFindV24string_lookup_85/hash_table_Lookup/LookupTableFindV2:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
<
__inference__creator_2043612
identity??
hash_table}

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	1653794*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
<
__inference__creator_2043645
identity??
hash_table}

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	1653906*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?

?
J__inference_embedding_238_layer_call_and_return_conditional_losses_2043468

inputs	*
embedding_lookup_2043462:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2043462inputs",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/2043462*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@embedding_lookup/2043462*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?t
?
E__inference_model_29_layer_call_and_return_conditional_losses_2042937	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_yearsE
Astring_lookup_85_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_85_hash_table_lookup_lookuptablefindv2_default_value	E
Astring_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_84_hash_table_lookup_lookuptablefindv2_default_value	G
Cinteger_lookup_130_hash_table_lookup_lookuptablefindv2_table_handleH
Dinteger_lookup_130_hash_table_lookup_lookuptablefindv2_default_value	'
embedding_238_2042871:'
embedding_237_2042874:'
embedding_236_2042877:
normalization_348_sub_y
normalization_348_sqrt_x
normalization_349_sub_y
normalization_349_sqrt_x
normalization_350_sub_y
normalization_350_sqrt_x
normalization_351_sub_y
normalization_351_sqrt_x
normalization_352_sub_y
normalization_352_sqrt_x
normalization_353_sub_y
normalization_353_sqrt_x"
dense_59_2042926:)

dense_59_2042928:
"
dense_60_2042931:

dense_60_2042933:
identity?? dense_59/StatefulPartitionedCall? dense_60/StatefulPartitionedCall?%embedding_236/StatefulPartitionedCall?%embedding_237/StatefulPartitionedCall?%embedding_238/StatefulPartitionedCall?6integer_lookup_130/hash_table_Lookup/LookupTableFindV2?4string_lookup_84/hash_table_Lookup/LookupTableFindV2?4string_lookup_85/hash_table_Lookup/LookupTableFindV2?
4string_lookup_85/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_85_hash_table_lookup_lookuptablefindv2_table_handlestorey_rangeBstring_lookup_85_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_85/hash_table_Lookup/LookupTableFindV2?
string_lookup_85/IdentityIdentity=string_lookup_85/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_85/Identity?
4string_lookup_84/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleflat_model_typeBstring_lookup_84_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_84/hash_table_Lookup/LookupTableFindV2?
string_lookup_84/IdentityIdentity=string_lookup_84/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_84/Identity?
6integer_lookup_130/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Cinteger_lookup_130_hash_table_lookup_lookuptablefindv2_table_handlemonthDinteger_lookup_130_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????28
6integer_lookup_130/hash_table_Lookup/LookupTableFindV2?
integer_lookup_130/IdentityIdentity?integer_lookup_130/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup_130/Identity?
%embedding_238/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_85/Identity:output:0embedding_238_2042871*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_238_layer_call_and_return_conditional_losses_20422952'
%embedding_238/StatefulPartitionedCall?
%embedding_237/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_84/Identity:output:0embedding_237_2042874*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_237_layer_call_and_return_conditional_losses_20423082'
%embedding_237/StatefulPartitionedCall?
%embedding_236/StatefulPartitionedCallStatefulPartitionedCall$integer_lookup_130/Identity:output:0embedding_236_2042877*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_236_layer_call_and_return_conditional_losses_20423212'
%embedding_236/StatefulPartitionedCall?
flatten_45/PartitionedCallPartitionedCall.embedding_236/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_45_layer_call_and_return_conditional_losses_20423312
flatten_45/PartitionedCall?
flatten_46/PartitionedCallPartitionedCall.embedding_237/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_46_layer_call_and_return_conditional_losses_20423392
flatten_46/PartitionedCall?
flatten_47/PartitionedCallPartitionedCall.embedding_238/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_47_layer_call_and_return_conditional_losses_20423472
flatten_47/PartitionedCall?
normalization_348/subSubfloor_area_sqmnormalization_348_sub_y*
T0*'
_output_shapes
:?????????2
normalization_348/sub{
normalization_348/SqrtSqrtnormalization_348_sqrt_x*
T0*
_output_shapes

:2
normalization_348/Sqrt
normalization_348/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_348/Maximum/y?
normalization_348/MaximumMaximumnormalization_348/Sqrt:y:0$normalization_348/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_348/Maximum?
normalization_348/truedivRealDivnormalization_348/sub:z:0normalization_348/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_348/truediv?
normalization_349/subSubdist_to_nearest_stnnormalization_349_sub_y*
T0*'
_output_shapes
:?????????2
normalization_349/sub{
normalization_349/SqrtSqrtnormalization_349_sqrt_x*
T0*
_output_shapes

:2
normalization_349/Sqrt
normalization_349/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_349/Maximum/y?
normalization_349/MaximumMaximumnormalization_349/Sqrt:y:0$normalization_349/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_349/Maximum?
normalization_349/truedivRealDivnormalization_349/sub:z:0normalization_349/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_349/truediv?
normalization_350/subSubdist_to_dhobynormalization_350_sub_y*
T0*'
_output_shapes
:?????????2
normalization_350/sub{
normalization_350/SqrtSqrtnormalization_350_sqrt_x*
T0*
_output_shapes

:2
normalization_350/Sqrt
normalization_350/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_350/Maximum/y?
normalization_350/MaximumMaximumnormalization_350/Sqrt:y:0$normalization_350/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_350/Maximum?
normalization_350/truedivRealDivnormalization_350/sub:z:0normalization_350/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_350/truediv?
normalization_351/subSubdegree_centralitynormalization_351_sub_y*
T0*'
_output_shapes
:?????????2
normalization_351/sub{
normalization_351/SqrtSqrtnormalization_351_sqrt_x*
T0*
_output_shapes

:2
normalization_351/Sqrt
normalization_351/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_351/Maximum/y?
normalization_351/MaximumMaximumnormalization_351/Sqrt:y:0$normalization_351/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_351/Maximum?
normalization_351/truedivRealDivnormalization_351/sub:z:0normalization_351/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_351/truediv?
normalization_352/subSubeigenvector_centralitynormalization_352_sub_y*
T0*'
_output_shapes
:?????????2
normalization_352/sub{
normalization_352/SqrtSqrtnormalization_352_sqrt_x*
T0*
_output_shapes

:2
normalization_352/Sqrt
normalization_352/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_352/Maximum/y?
normalization_352/MaximumMaximumnormalization_352/Sqrt:y:0$normalization_352/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_352/Maximum?
normalization_352/truedivRealDivnormalization_352/sub:z:0normalization_352/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_352/truediv?
normalization_353/subSubremaining_lease_yearsnormalization_353_sub_y*
T0*'
_output_shapes
:?????????2
normalization_353/sub{
normalization_353/SqrtSqrtnormalization_353_sqrt_x*
T0*
_output_shapes

:2
normalization_353/Sqrt
normalization_353/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_353/Maximum/y?
normalization_353/MaximumMaximumnormalization_353/Sqrt:y:0$normalization_353/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_353/Maximum?
normalization_353/truedivRealDivnormalization_353/sub:z:0normalization_353/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_353/truediv?
concatenate_58/PartitionedCallPartitionedCall#flatten_45/PartitionedCall:output:0#flatten_46/PartitionedCall:output:0#flatten_47/PartitionedCall:output:0normalization_348/truediv:z:0normalization_349/truediv:z:0normalization_350/truediv:z:0normalization_351/truediv:z:0normalization_352/truediv:z:0normalization_353/truediv:z:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????)* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_concatenate_58_layer_call_and_return_conditional_losses_20424052 
concatenate_58/PartitionedCall?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall'concatenate_58/PartitionedCall:output:0dense_59_2042926dense_59_2042928*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_20424182"
 dense_59/StatefulPartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall)dense_59/StatefulPartitionedCall:output:0dense_60_2042931dense_60_2042933*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_60_layer_call_and_return_conditional_losses_20424342"
 dense_60/StatefulPartitionedCall?
IdentityIdentity)dense_60/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_59/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall&^embedding_236/StatefulPartitionedCall&^embedding_237/StatefulPartitionedCall&^embedding_238/StatefulPartitionedCall7^integer_lookup_130/hash_table_Lookup/LookupTableFindV25^string_lookup_84/hash_table_Lookup/LookupTableFindV25^string_lookup_85/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : ::::::::::::: : : : 2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2N
%embedding_236/StatefulPartitionedCall%embedding_236/StatefulPartitionedCall2N
%embedding_237/StatefulPartitionedCall%embedding_237/StatefulPartitionedCall2N
%embedding_238/StatefulPartitionedCall%embedding_238/StatefulPartitionedCall2p
6integer_lookup_130/hash_table_Lookup/LookupTableFindV26integer_lookup_130/hash_table_Lookup/LookupTableFindV22l
4string_lookup_84/hash_table_Lookup/LookupTableFindV24string_lookup_84/hash_table_Lookup/LookupTableFindV22l
4string_lookup_85/hash_table_Lookup/LookupTableFindV24string_lookup_85/hash_table_Lookup/LookupTableFindV2:N J
'
_output_shapes
:?????????

_user_specified_namemonth:XT
'
_output_shapes
:?????????
)
_user_specified_nameflat_model_type:UQ
'
_output_shapes
:?????????
&
_user_specified_namestorey_range:WS
'
_output_shapes
:?????????
(
_user_specified_namefloor_area_sqm:\X
'
_output_shapes
:?????????
-
_user_specified_namedist_to_nearest_stn:VR
'
_output_shapes
:?????????
'
_user_specified_namedist_to_dhoby:ZV
'
_output_shapes
:?????????
+
_user_specified_namedegree_centrality:_[
'
_output_shapes
:?????????
0
_user_specified_nameeigenvector_centrality:^Z
'
_output_shapes
:?????????
/
_user_specified_nameremaining_lease_years:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?	
?
__inference_restore_fn_2043754
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 24
2MutableHashTable_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
0
 __inference__initializer_2043635
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
J__inference_embedding_236_layer_call_and_return_conditional_losses_2042321

inputs	*
embedding_lookup_2042315:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2042315inputs",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/2042315*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@embedding_lookup/2042315*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_model_29_layer_call_fn_2042494	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_years
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20:)


unknown_21:


unknown_22:


unknown_23:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmonthflat_model_typestorey_rangefloor_area_sqmdist_to_nearest_stndist_to_dhobydegree_centralityeigenvector_centralityremaining_lease_yearsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*-
Tin&
$2"				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	 !*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_29_layer_call_and_return_conditional_losses_20424412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : ::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namemonth:XT
'
_output_shapes
:?????????
)
_user_specified_nameflat_model_type:UQ
'
_output_shapes
:?????????
&
_user_specified_namestorey_range:WS
'
_output_shapes
:?????????
(
_user_specified_namefloor_area_sqm:\X
'
_output_shapes
:?????????
-
_user_specified_namedist_to_nearest_stn:VR
'
_output_shapes
:?????????
'
_user_specified_namedist_to_dhoby:ZV
'
_output_shapes
:?????????
+
_user_specified_namedegree_centrality:_[
'
_output_shapes
:?????????
0
_user_specified_nameeigenvector_centrality:^Z
'
_output_shapes
:?????????
/
_user_specified_nameremaining_lease_years:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
c
G__inference_flatten_47_layer_call_and_return_conditional_losses_2042347

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
,
__inference_<lambda>_2043780
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
H
,__inference_flatten_46_layer_call_fn_2043497

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_46_layer_call_and_return_conditional_losses_20423392
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
.
__inference__destroyer_2043640
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
*__inference_dense_59_layer_call_fn_2043555

inputs
unknown:)

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_20424182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????): : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?,
?
__inference_adapt_step_1661455
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNexts
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
?
 __inference__initializer_2043653:
6key_value_init1653905_lookuptableimportv2_table_handle2
.key_value_init1653905_lookuptableimportv2_keys4
0key_value_init1653905_lookuptableimportv2_values	
identity??)key_value_init1653905/LookupTableImportV2?
)key_value_init1653905/LookupTableImportV2LookupTableImportV26key_value_init1653905_lookuptableimportv2_table_handle.key_value_init1653905_lookuptableimportv2_keys0key_value_init1653905_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2+
)key_value_init1653905/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityz
NoOpNoOp*^key_value_init1653905/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init1653905/LookupTableImportV2)key_value_init1653905/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?,
?
__inference_adapt_step_1661690
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNexts
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
?
/__inference_embedding_237_layer_call_fn_2043459

inputs	
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_237_layer_call_and_return_conditional_losses_20423082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_2043775:
6key_value_init1653793_lookuptableimportv2_table_handle2
.key_value_init1653793_lookuptableimportv2_keys4
0key_value_init1653793_lookuptableimportv2_values	
identity??)key_value_init1653793/LookupTableImportV2?
)key_value_init1653793/LookupTableImportV2LookupTableImportV26key_value_init1653793_lookuptableimportv2_table_handle.key_value_init1653793_lookuptableimportv2_keys0key_value_init1653793_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2+
)key_value_init1653793/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityz
NoOpNoOp*^key_value_init1653793/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :+:+2V
)key_value_init1653793/LookupTableImportV2)key_value_init1653793/LookupTableImportV2: 

_output_shapes
:+: 

_output_shapes
:+
?
c
G__inference_flatten_46_layer_call_and_return_conditional_losses_2042339

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
__inference__creator_2043630
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_1653717*
value_dtype0	2
MutableHashTablei
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identitya
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
?
/__inference_embedding_236_layer_call_fn_2043443

inputs	
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_236_layer_call_and_return_conditional_losses_20423212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_45_layer_call_and_return_conditional_losses_2042331

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_60_layer_call_fn_2043574

inputs
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_60_layer_call_and_return_conditional_losses_20424342
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
.
__inference__destroyer_2043673
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?s
?
E__inference_model_29_layer_call_and_return_conditional_losses_2042441

inputs	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8E
Astring_lookup_85_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_85_hash_table_lookup_lookuptablefindv2_default_value	E
Astring_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_84_hash_table_lookup_lookuptablefindv2_default_value	G
Cinteger_lookup_130_hash_table_lookup_lookuptablefindv2_table_handleH
Dinteger_lookup_130_hash_table_lookup_lookuptablefindv2_default_value	'
embedding_238_2042296:'
embedding_237_2042309:'
embedding_236_2042322:
normalization_348_sub_y
normalization_348_sqrt_x
normalization_349_sub_y
normalization_349_sqrt_x
normalization_350_sub_y
normalization_350_sqrt_x
normalization_351_sub_y
normalization_351_sqrt_x
normalization_352_sub_y
normalization_352_sqrt_x
normalization_353_sub_y
normalization_353_sqrt_x"
dense_59_2042419:)

dense_59_2042421:
"
dense_60_2042435:

dense_60_2042437:
identity?? dense_59/StatefulPartitionedCall? dense_60/StatefulPartitionedCall?%embedding_236/StatefulPartitionedCall?%embedding_237/StatefulPartitionedCall?%embedding_238/StatefulPartitionedCall?6integer_lookup_130/hash_table_Lookup/LookupTableFindV2?4string_lookup_84/hash_table_Lookup/LookupTableFindV2?4string_lookup_85/hash_table_Lookup/LookupTableFindV2?
4string_lookup_85/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_85_hash_table_lookup_lookuptablefindv2_table_handleinputs_2Bstring_lookup_85_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_85/hash_table_Lookup/LookupTableFindV2?
string_lookup_85/IdentityIdentity=string_lookup_85/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_85/Identity?
4string_lookup_84/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleinputs_1Bstring_lookup_84_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_84/hash_table_Lookup/LookupTableFindV2?
string_lookup_84/IdentityIdentity=string_lookup_84/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_84/Identity?
6integer_lookup_130/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Cinteger_lookup_130_hash_table_lookup_lookuptablefindv2_table_handleinputsDinteger_lookup_130_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????28
6integer_lookup_130/hash_table_Lookup/LookupTableFindV2?
integer_lookup_130/IdentityIdentity?integer_lookup_130/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup_130/Identity?
%embedding_238/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_85/Identity:output:0embedding_238_2042296*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_238_layer_call_and_return_conditional_losses_20422952'
%embedding_238/StatefulPartitionedCall?
%embedding_237/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_84/Identity:output:0embedding_237_2042309*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_237_layer_call_and_return_conditional_losses_20423082'
%embedding_237/StatefulPartitionedCall?
%embedding_236/StatefulPartitionedCallStatefulPartitionedCall$integer_lookup_130/Identity:output:0embedding_236_2042322*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_236_layer_call_and_return_conditional_losses_20423212'
%embedding_236/StatefulPartitionedCall?
flatten_45/PartitionedCallPartitionedCall.embedding_236/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_45_layer_call_and_return_conditional_losses_20423312
flatten_45/PartitionedCall?
flatten_46/PartitionedCallPartitionedCall.embedding_237/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_46_layer_call_and_return_conditional_losses_20423392
flatten_46/PartitionedCall?
flatten_47/PartitionedCallPartitionedCall.embedding_238/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_47_layer_call_and_return_conditional_losses_20423472
flatten_47/PartitionedCall?
normalization_348/subSubinputs_3normalization_348_sub_y*
T0*'
_output_shapes
:?????????2
normalization_348/sub{
normalization_348/SqrtSqrtnormalization_348_sqrt_x*
T0*
_output_shapes

:2
normalization_348/Sqrt
normalization_348/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_348/Maximum/y?
normalization_348/MaximumMaximumnormalization_348/Sqrt:y:0$normalization_348/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_348/Maximum?
normalization_348/truedivRealDivnormalization_348/sub:z:0normalization_348/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_348/truediv?
normalization_349/subSubinputs_4normalization_349_sub_y*
T0*'
_output_shapes
:?????????2
normalization_349/sub{
normalization_349/SqrtSqrtnormalization_349_sqrt_x*
T0*
_output_shapes

:2
normalization_349/Sqrt
normalization_349/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_349/Maximum/y?
normalization_349/MaximumMaximumnormalization_349/Sqrt:y:0$normalization_349/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_349/Maximum?
normalization_349/truedivRealDivnormalization_349/sub:z:0normalization_349/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_349/truediv?
normalization_350/subSubinputs_5normalization_350_sub_y*
T0*'
_output_shapes
:?????????2
normalization_350/sub{
normalization_350/SqrtSqrtnormalization_350_sqrt_x*
T0*
_output_shapes

:2
normalization_350/Sqrt
normalization_350/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_350/Maximum/y?
normalization_350/MaximumMaximumnormalization_350/Sqrt:y:0$normalization_350/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_350/Maximum?
normalization_350/truedivRealDivnormalization_350/sub:z:0normalization_350/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_350/truediv?
normalization_351/subSubinputs_6normalization_351_sub_y*
T0*'
_output_shapes
:?????????2
normalization_351/sub{
normalization_351/SqrtSqrtnormalization_351_sqrt_x*
T0*
_output_shapes

:2
normalization_351/Sqrt
normalization_351/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_351/Maximum/y?
normalization_351/MaximumMaximumnormalization_351/Sqrt:y:0$normalization_351/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_351/Maximum?
normalization_351/truedivRealDivnormalization_351/sub:z:0normalization_351/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_351/truediv?
normalization_352/subSubinputs_7normalization_352_sub_y*
T0*'
_output_shapes
:?????????2
normalization_352/sub{
normalization_352/SqrtSqrtnormalization_352_sqrt_x*
T0*
_output_shapes

:2
normalization_352/Sqrt
normalization_352/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_352/Maximum/y?
normalization_352/MaximumMaximumnormalization_352/Sqrt:y:0$normalization_352/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_352/Maximum?
normalization_352/truedivRealDivnormalization_352/sub:z:0normalization_352/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_352/truediv?
normalization_353/subSubinputs_8normalization_353_sub_y*
T0*'
_output_shapes
:?????????2
normalization_353/sub{
normalization_353/SqrtSqrtnormalization_353_sqrt_x*
T0*
_output_shapes

:2
normalization_353/Sqrt
normalization_353/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_353/Maximum/y?
normalization_353/MaximumMaximumnormalization_353/Sqrt:y:0$normalization_353/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_353/Maximum?
normalization_353/truedivRealDivnormalization_353/sub:z:0normalization_353/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_353/truediv?
concatenate_58/PartitionedCallPartitionedCall#flatten_45/PartitionedCall:output:0#flatten_46/PartitionedCall:output:0#flatten_47/PartitionedCall:output:0normalization_348/truediv:z:0normalization_349/truediv:z:0normalization_350/truediv:z:0normalization_351/truediv:z:0normalization_352/truediv:z:0normalization_353/truediv:z:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????)* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_concatenate_58_layer_call_and_return_conditional_losses_20424052 
concatenate_58/PartitionedCall?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall'concatenate_58/PartitionedCall:output:0dense_59_2042419dense_59_2042421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_20424182"
 dense_59/StatefulPartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall)dense_59/StatefulPartitionedCall:output:0dense_60_2042435dense_60_2042437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_60_layer_call_and_return_conditional_losses_20424342"
 dense_60/StatefulPartitionedCall?
IdentityIdentity)dense_60/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_59/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall&^embedding_236/StatefulPartitionedCall&^embedding_237/StatefulPartitionedCall&^embedding_238/StatefulPartitionedCall7^integer_lookup_130/hash_table_Lookup/LookupTableFindV25^string_lookup_84/hash_table_Lookup/LookupTableFindV25^string_lookup_85/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : ::::::::::::: : : : 2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2N
%embedding_236/StatefulPartitionedCall%embedding_236/StatefulPartitionedCall2N
%embedding_237/StatefulPartitionedCall%embedding_237/StatefulPartitionedCall2N
%embedding_238/StatefulPartitionedCall%embedding_238/StatefulPartitionedCall2p
6integer_lookup_130/hash_table_Lookup/LookupTableFindV26integer_lookup_130/hash_table_Lookup/LookupTableFindV22l
4string_lookup_84/hash_table_Lookup/LookupTableFindV24string_lookup_84/hash_table_Lookup/LookupTableFindV22l
4string_lookup_85/hash_table_Lookup/LookupTableFindV24string_lookup_85/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
*__inference_model_29_layer_call_fn_2042848	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_years
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20:)


unknown_21:


unknown_22:


unknown_23:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmonthflat_model_typestorey_rangefloor_area_sqmdist_to_nearest_stndist_to_dhobydegree_centralityeigenvector_centralityremaining_lease_yearsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*-
Tin&
$2"				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	 !*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_29_layer_call_and_return_conditional_losses_20427322
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : ::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namemonth:XT
'
_output_shapes
:?????????
)
_user_specified_nameflat_model_type:UQ
'
_output_shapes
:?????????
&
_user_specified_namestorey_range:WS
'
_output_shapes
:?????????
(
_user_specified_namefloor_area_sqm:\X
'
_output_shapes
:?????????
-
_user_specified_namedist_to_nearest_stn:VR
'
_output_shapes
:?????????
'
_user_specified_namedist_to_dhoby:ZV
'
_output_shapes
:?????????
+
_user_specified_namedegree_centrality:_[
'
_output_shapes
:?????????
0
_user_specified_nameeigenvector_centrality:^Z
'
_output_shapes
:?????????
/
_user_specified_nameremaining_lease_years:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
H
__inference__creator_2043597
identity:	 ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_nametable_1653605*
value_dtype0	2
MutableHashTablei
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identitya
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?,
?
__inference_adapt_step_1661502
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNexts
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?

?
J__inference_embedding_236_layer_call_and_return_conditional_losses_2043436

inputs	*
embedding_lookup_2043430:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2043430inputs",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/2043430*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@embedding_lookup/2043430*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
 __inference__initializer_2043620:
6key_value_init1653793_lookuptableimportv2_table_handle2
.key_value_init1653793_lookuptableimportv2_keys4
0key_value_init1653793_lookuptableimportv2_values	
identity??)key_value_init1653793/LookupTableImportV2?
)key_value_init1653793/LookupTableImportV2LookupTableImportV26key_value_init1653793_lookuptableimportv2_table_handle.key_value_init1653793_lookuptableimportv2_keys0key_value_init1653793_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2+
)key_value_init1653793/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityz
NoOpNoOp*^key_value_init1653793/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :+:+2V
)key_value_init1653793/LookupTableImportV2)key_value_init1653793/LookupTableImportV2: 

_output_shapes
:+: 

_output_shapes
:+
?t
?
E__inference_model_29_layer_call_and_return_conditional_losses_2043026	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_yearsE
Astring_lookup_85_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_85_hash_table_lookup_lookuptablefindv2_default_value	E
Astring_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_84_hash_table_lookup_lookuptablefindv2_default_value	G
Cinteger_lookup_130_hash_table_lookup_lookuptablefindv2_table_handleH
Dinteger_lookup_130_hash_table_lookup_lookuptablefindv2_default_value	'
embedding_238_2042960:'
embedding_237_2042963:'
embedding_236_2042966:
normalization_348_sub_y
normalization_348_sqrt_x
normalization_349_sub_y
normalization_349_sqrt_x
normalization_350_sub_y
normalization_350_sqrt_x
normalization_351_sub_y
normalization_351_sqrt_x
normalization_352_sub_y
normalization_352_sqrt_x
normalization_353_sub_y
normalization_353_sqrt_x"
dense_59_2043015:)

dense_59_2043017:
"
dense_60_2043020:

dense_60_2043022:
identity?? dense_59/StatefulPartitionedCall? dense_60/StatefulPartitionedCall?%embedding_236/StatefulPartitionedCall?%embedding_237/StatefulPartitionedCall?%embedding_238/StatefulPartitionedCall?6integer_lookup_130/hash_table_Lookup/LookupTableFindV2?4string_lookup_84/hash_table_Lookup/LookupTableFindV2?4string_lookup_85/hash_table_Lookup/LookupTableFindV2?
4string_lookup_85/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_85_hash_table_lookup_lookuptablefindv2_table_handlestorey_rangeBstring_lookup_85_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_85/hash_table_Lookup/LookupTableFindV2?
string_lookup_85/IdentityIdentity=string_lookup_85/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_85/Identity?
4string_lookup_84/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleflat_model_typeBstring_lookup_84_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_84/hash_table_Lookup/LookupTableFindV2?
string_lookup_84/IdentityIdentity=string_lookup_84/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_84/Identity?
6integer_lookup_130/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Cinteger_lookup_130_hash_table_lookup_lookuptablefindv2_table_handlemonthDinteger_lookup_130_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????28
6integer_lookup_130/hash_table_Lookup/LookupTableFindV2?
integer_lookup_130/IdentityIdentity?integer_lookup_130/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup_130/Identity?
%embedding_238/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_85/Identity:output:0embedding_238_2042960*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_238_layer_call_and_return_conditional_losses_20422952'
%embedding_238/StatefulPartitionedCall?
%embedding_237/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_84/Identity:output:0embedding_237_2042963*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_237_layer_call_and_return_conditional_losses_20423082'
%embedding_237/StatefulPartitionedCall?
%embedding_236/StatefulPartitionedCallStatefulPartitionedCall$integer_lookup_130/Identity:output:0embedding_236_2042966*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_236_layer_call_and_return_conditional_losses_20423212'
%embedding_236/StatefulPartitionedCall?
flatten_45/PartitionedCallPartitionedCall.embedding_236/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_45_layer_call_and_return_conditional_losses_20423312
flatten_45/PartitionedCall?
flatten_46/PartitionedCallPartitionedCall.embedding_237/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_46_layer_call_and_return_conditional_losses_20423392
flatten_46/PartitionedCall?
flatten_47/PartitionedCallPartitionedCall.embedding_238/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_47_layer_call_and_return_conditional_losses_20423472
flatten_47/PartitionedCall?
normalization_348/subSubfloor_area_sqmnormalization_348_sub_y*
T0*'
_output_shapes
:?????????2
normalization_348/sub{
normalization_348/SqrtSqrtnormalization_348_sqrt_x*
T0*
_output_shapes

:2
normalization_348/Sqrt
normalization_348/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_348/Maximum/y?
normalization_348/MaximumMaximumnormalization_348/Sqrt:y:0$normalization_348/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_348/Maximum?
normalization_348/truedivRealDivnormalization_348/sub:z:0normalization_348/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_348/truediv?
normalization_349/subSubdist_to_nearest_stnnormalization_349_sub_y*
T0*'
_output_shapes
:?????????2
normalization_349/sub{
normalization_349/SqrtSqrtnormalization_349_sqrt_x*
T0*
_output_shapes

:2
normalization_349/Sqrt
normalization_349/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_349/Maximum/y?
normalization_349/MaximumMaximumnormalization_349/Sqrt:y:0$normalization_349/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_349/Maximum?
normalization_349/truedivRealDivnormalization_349/sub:z:0normalization_349/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_349/truediv?
normalization_350/subSubdist_to_dhobynormalization_350_sub_y*
T0*'
_output_shapes
:?????????2
normalization_350/sub{
normalization_350/SqrtSqrtnormalization_350_sqrt_x*
T0*
_output_shapes

:2
normalization_350/Sqrt
normalization_350/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_350/Maximum/y?
normalization_350/MaximumMaximumnormalization_350/Sqrt:y:0$normalization_350/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_350/Maximum?
normalization_350/truedivRealDivnormalization_350/sub:z:0normalization_350/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_350/truediv?
normalization_351/subSubdegree_centralitynormalization_351_sub_y*
T0*'
_output_shapes
:?????????2
normalization_351/sub{
normalization_351/SqrtSqrtnormalization_351_sqrt_x*
T0*
_output_shapes

:2
normalization_351/Sqrt
normalization_351/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_351/Maximum/y?
normalization_351/MaximumMaximumnormalization_351/Sqrt:y:0$normalization_351/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_351/Maximum?
normalization_351/truedivRealDivnormalization_351/sub:z:0normalization_351/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_351/truediv?
normalization_352/subSubeigenvector_centralitynormalization_352_sub_y*
T0*'
_output_shapes
:?????????2
normalization_352/sub{
normalization_352/SqrtSqrtnormalization_352_sqrt_x*
T0*
_output_shapes

:2
normalization_352/Sqrt
normalization_352/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_352/Maximum/y?
normalization_352/MaximumMaximumnormalization_352/Sqrt:y:0$normalization_352/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_352/Maximum?
normalization_352/truedivRealDivnormalization_352/sub:z:0normalization_352/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_352/truediv?
normalization_353/subSubremaining_lease_yearsnormalization_353_sub_y*
T0*'
_output_shapes
:?????????2
normalization_353/sub{
normalization_353/SqrtSqrtnormalization_353_sqrt_x*
T0*
_output_shapes

:2
normalization_353/Sqrt
normalization_353/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_353/Maximum/y?
normalization_353/MaximumMaximumnormalization_353/Sqrt:y:0$normalization_353/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_353/Maximum?
normalization_353/truedivRealDivnormalization_353/sub:z:0normalization_353/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_353/truediv?
concatenate_58/PartitionedCallPartitionedCall#flatten_45/PartitionedCall:output:0#flatten_46/PartitionedCall:output:0#flatten_47/PartitionedCall:output:0normalization_348/truediv:z:0normalization_349/truediv:z:0normalization_350/truediv:z:0normalization_351/truediv:z:0normalization_352/truediv:z:0normalization_353/truediv:z:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????)* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_concatenate_58_layer_call_and_return_conditional_losses_20424052 
concatenate_58/PartitionedCall?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall'concatenate_58/PartitionedCall:output:0dense_59_2043015dense_59_2043017*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_20424182"
 dense_59/StatefulPartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall)dense_59/StatefulPartitionedCall:output:0dense_60_2043020dense_60_2043022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_60_layer_call_and_return_conditional_losses_20424342"
 dense_60/StatefulPartitionedCall?
IdentityIdentity)dense_60/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_59/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall&^embedding_236/StatefulPartitionedCall&^embedding_237/StatefulPartitionedCall&^embedding_238/StatefulPartitionedCall7^integer_lookup_130/hash_table_Lookup/LookupTableFindV25^string_lookup_84/hash_table_Lookup/LookupTableFindV25^string_lookup_85/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : ::::::::::::: : : : 2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2N
%embedding_236/StatefulPartitionedCall%embedding_236/StatefulPartitionedCall2N
%embedding_237/StatefulPartitionedCall%embedding_237/StatefulPartitionedCall2N
%embedding_238/StatefulPartitionedCall%embedding_238/StatefulPartitionedCall2p
6integer_lookup_130/hash_table_Lookup/LookupTableFindV26integer_lookup_130/hash_table_Lookup/LookupTableFindV22l
4string_lookup_84/hash_table_Lookup/LookupTableFindV24string_lookup_84/hash_table_Lookup/LookupTableFindV22l
4string_lookup_85/hash_table_Lookup/LookupTableFindV24string_lookup_85/hash_table_Lookup/LookupTableFindV2:N J
'
_output_shapes
:?????????

_user_specified_namemonth:XT
'
_output_shapes
:?????????
)
_user_specified_nameflat_model_type:UQ
'
_output_shapes
:?????????
&
_user_specified_namestorey_range:WS
'
_output_shapes
:?????????
(
_user_specified_namefloor_area_sqm:\X
'
_output_shapes
:?????????
-
_user_specified_namedist_to_nearest_stn:VR
'
_output_shapes
:?????????
'
_user_specified_namedist_to_dhoby:ZV
'
_output_shapes
:?????????
+
_user_specified_namedegree_centrality:_[
'
_output_shapes
:?????????
0
_user_specified_nameeigenvector_centrality:^Z
'
_output_shapes
:?????????
/
_user_specified_nameremaining_lease_years:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
H
,__inference_flatten_47_layer_call_fn_2043508

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_47_layer_call_and_return_conditional_losses_20423472
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
__inference_adapt_step_1661596
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNexts
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
?
__inference_save_fn_2043692
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2	

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0	*
Tvalues0	*
_output_shapes

::2A
?MutableHashTable_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1Q
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const\

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0	*
_output_shapes
:2

Identity_2W

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1^

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:2

Identity_5?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?n
?
 __inference__traced_save_2044028
file_prefixJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2	L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	L
Hsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2N
Jsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2_1	L
Hsavev2_mutablehashtable_2_lookup_table_export_values_lookuptableexportv2N
Jsavev2_mutablehashtable_2_lookup_table_export_values_lookuptableexportv2_1	7
3savev2_embedding_236_embeddings_read_readvariableop7
3savev2_embedding_237_embeddings_read_readvariableop7
3savev2_embedding_238_embeddings_read_readvariableop#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	%
!savev2_mean_1_read_readvariableop)
%savev2_variance_1_read_readvariableop&
"savev2_count_1_read_readvariableop	%
!savev2_mean_2_read_readvariableop)
%savev2_variance_2_read_readvariableop&
"savev2_count_2_read_readvariableop	%
!savev2_mean_3_read_readvariableop)
%savev2_variance_3_read_readvariableop&
"savev2_count_3_read_readvariableop	%
!savev2_mean_4_read_readvariableop)
%savev2_variance_4_read_readvariableop&
"savev2_count_4_read_readvariableop	%
!savev2_mean_5_read_readvariableop)
%savev2_variance_5_read_readvariableop&
"savev2_count_5_read_readvariableop	.
*savev2_dense_59_kernel_read_readvariableop,
(savev2_dense_59_bias_read_readvariableop.
*savev2_dense_60_kernel_read_readvariableop,
(savev2_dense_60_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_6_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_7_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_8_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_9_read_readvariableop>
:savev2_adam_embedding_236_embeddings_m_read_readvariableop>
:savev2_adam_embedding_237_embeddings_m_read_readvariableop>
:savev2_adam_embedding_238_embeddings_m_read_readvariableop5
1savev2_adam_dense_59_kernel_m_read_readvariableop3
/savev2_adam_dense_59_bias_m_read_readvariableop5
1savev2_adam_dense_60_kernel_m_read_readvariableop3
/savev2_adam_dense_60_bias_m_read_readvariableop>
:savev2_adam_embedding_236_embeddings_v_read_readvariableop>
:savev2_adam_embedding_237_embeddings_v_read_readvariableop>
:savev2_adam_embedding_238_embeddings_v_read_readvariableop5
1savev2_adam_dense_59_kernel_v_read_readvariableop3
/savev2_adam_dense_59_bias_v_read_readvariableop5
1savev2_adam_dense_60_kernel_v_read_readvariableop3
/savev2_adam_dense_60_bias_v_read_readvariableop
savev2_const_24

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B?;B8layer_with_weights-0/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-0/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-1/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-1/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-2/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-2/token_counts/.ATTRIBUTES/table-valuesB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-7/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-8/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-9/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-10/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-11/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/count/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1Hsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2Jsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2_1Hsavev2_mutablehashtable_2_lookup_table_export_values_lookuptableexportv2Jsavev2_mutablehashtable_2_lookup_table_export_values_lookuptableexportv2_13savev2_embedding_236_embeddings_read_readvariableop3savev2_embedding_237_embeddings_read_readvariableop3savev2_embedding_238_embeddings_read_readvariableopsavev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop!savev2_mean_1_read_readvariableop%savev2_variance_1_read_readvariableop"savev2_count_1_read_readvariableop!savev2_mean_2_read_readvariableop%savev2_variance_2_read_readvariableop"savev2_count_2_read_readvariableop!savev2_mean_3_read_readvariableop%savev2_variance_3_read_readvariableop"savev2_count_3_read_readvariableop!savev2_mean_4_read_readvariableop%savev2_variance_4_read_readvariableop"savev2_count_4_read_readvariableop!savev2_mean_5_read_readvariableop%savev2_variance_5_read_readvariableop"savev2_count_5_read_readvariableop*savev2_dense_59_kernel_read_readvariableop(savev2_dense_59_bias_read_readvariableop*savev2_dense_60_kernel_read_readvariableop(savev2_dense_60_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop"savev2_count_6_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_7_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_8_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_9_read_readvariableop:savev2_adam_embedding_236_embeddings_m_read_readvariableop:savev2_adam_embedding_237_embeddings_m_read_readvariableop:savev2_adam_embedding_238_embeddings_m_read_readvariableop1savev2_adam_dense_59_kernel_m_read_readvariableop/savev2_adam_dense_59_bias_m_read_readvariableop1savev2_adam_dense_60_kernel_m_read_readvariableop/savev2_adam_dense_60_bias_m_read_readvariableop:savev2_adam_embedding_236_embeddings_v_read_readvariableop:savev2_adam_embedding_237_embeddings_v_read_readvariableop:savev2_adam_embedding_238_embeddings_v_read_readvariableop1savev2_adam_dense_59_kernel_v_read_readvariableop/savev2_adam_dense_59_bias_v_read_readvariableop1savev2_adam_dense_60_kernel_v_read_readvariableop/savev2_adam_dense_60_bias_v_read_readvariableopsavev2_const_24"/device:CPU:0*
_output_shapes
 *I
dtypes?
=2;											2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::::::: ::: ::: ::: ::: ::: :)
:
:
:: : : : : : : : : : : : : ::::)
:
:
:::::)
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::$	 

_output_shapes

:: 


_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :$ 

_output_shapes

:)
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
:: 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :$- 

_output_shapes

::$. 

_output_shapes

::$/ 

_output_shapes

::$0 

_output_shapes

:)
: 1

_output_shapes
:
:$2 

_output_shapes

:
: 3

_output_shapes
::$4 

_output_shapes

::$5 

_output_shapes

::$6 

_output_shapes

::$7 

_output_shapes

:)
: 8

_output_shapes
:
:$9 

_output_shapes

:
: :

_output_shapes
::;

_output_shapes
: 
?
c
G__inference_flatten_47_layer_call_and_return_conditional_losses_2043503

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_45_layer_call_fn_2043486

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_45_layer_call_and_return_conditional_losses_20423312
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_adapt_step_1661299
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
2	2
IteratorGetNextq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shape?
ReshapeReshapeIteratorGetNext:components:0Reshape/shape:output:0*
T0	*#
_output_shapes
:?????????2	
Reshape?
UniqueWithCountsUniqueWithCountsReshape:output:0*
T0	*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	2
UniqueWithCounts?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0	*

Tout0	*
_output_shapes
:2*
(None_lookup_table_find/LookupTableFindV2?
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2
add?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0	*

Tout0	*
_output_shapes
 2.
,None_lookup_table_insert/LookupTableInsertV2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?

?
J__inference_embedding_237_layer_call_and_return_conditional_losses_2043452

inputs	*
embedding_lookup_2043446:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2043446inputs",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/2043446*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@embedding_lookup/2043446*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_model_29_layer_call_fn_2043364
inputs_0	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20:)


unknown_21:


unknown_22:


unknown_23:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*-
Tin&
$2"				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	 !*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_29_layer_call_and_return_conditional_losses_20424412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : ::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
??
?
E__inference_model_29_layer_call_and_return_conditional_losses_2043199
inputs_0	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8E
Astring_lookup_85_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_85_hash_table_lookup_lookuptablefindv2_default_value	E
Astring_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_84_hash_table_lookup_lookuptablefindv2_default_value	G
Cinteger_lookup_130_hash_table_lookup_lookuptablefindv2_table_handleH
Dinteger_lookup_130_hash_table_lookup_lookuptablefindv2_default_value	8
&embedding_238_embedding_lookup_2043120:8
&embedding_237_embedding_lookup_2043125:8
&embedding_236_embedding_lookup_2043130:
normalization_348_sub_y
normalization_348_sqrt_x
normalization_349_sub_y
normalization_349_sqrt_x
normalization_350_sub_y
normalization_350_sqrt_x
normalization_351_sub_y
normalization_351_sqrt_x
normalization_352_sub_y
normalization_352_sqrt_x
normalization_353_sub_y
normalization_353_sqrt_x9
'dense_59_matmul_readvariableop_resource:)
6
(dense_59_biasadd_readvariableop_resource:
9
'dense_60_matmul_readvariableop_resource:
6
(dense_60_biasadd_readvariableop_resource:
identity??dense_59/BiasAdd/ReadVariableOp?dense_59/MatMul/ReadVariableOp?dense_60/BiasAdd/ReadVariableOp?dense_60/MatMul/ReadVariableOp?embedding_236/embedding_lookup?embedding_237/embedding_lookup?embedding_238/embedding_lookup?6integer_lookup_130/hash_table_Lookup/LookupTableFindV2?4string_lookup_84/hash_table_Lookup/LookupTableFindV2?4string_lookup_85/hash_table_Lookup/LookupTableFindV2?
4string_lookup_85/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_85_hash_table_lookup_lookuptablefindv2_table_handleinputs_2Bstring_lookup_85_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_85/hash_table_Lookup/LookupTableFindV2?
string_lookup_85/IdentityIdentity=string_lookup_85/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_85/Identity?
4string_lookup_84/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleinputs_1Bstring_lookup_84_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_84/hash_table_Lookup/LookupTableFindV2?
string_lookup_84/IdentityIdentity=string_lookup_84/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_84/Identity?
6integer_lookup_130/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Cinteger_lookup_130_hash_table_lookup_lookuptablefindv2_table_handleinputs_0Dinteger_lookup_130_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????28
6integer_lookup_130/hash_table_Lookup/LookupTableFindV2?
integer_lookup_130/IdentityIdentity?integer_lookup_130/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup_130/Identity?
embedding_238/embedding_lookupResourceGather&embedding_238_embedding_lookup_2043120"string_lookup_85/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*9
_class/
-+loc:@embedding_238/embedding_lookup/2043120*+
_output_shapes
:?????????*
dtype02 
embedding_238/embedding_lookup?
'embedding_238/embedding_lookup/IdentityIdentity'embedding_238/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*9
_class/
-+loc:@embedding_238/embedding_lookup/2043120*+
_output_shapes
:?????????2)
'embedding_238/embedding_lookup/Identity?
)embedding_238/embedding_lookup/Identity_1Identity0embedding_238/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2+
)embedding_238/embedding_lookup/Identity_1?
embedding_237/embedding_lookupResourceGather&embedding_237_embedding_lookup_2043125"string_lookup_84/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*9
_class/
-+loc:@embedding_237/embedding_lookup/2043125*+
_output_shapes
:?????????*
dtype02 
embedding_237/embedding_lookup?
'embedding_237/embedding_lookup/IdentityIdentity'embedding_237/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*9
_class/
-+loc:@embedding_237/embedding_lookup/2043125*+
_output_shapes
:?????????2)
'embedding_237/embedding_lookup/Identity?
)embedding_237/embedding_lookup/Identity_1Identity0embedding_237/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2+
)embedding_237/embedding_lookup/Identity_1?
embedding_236/embedding_lookupResourceGather&embedding_236_embedding_lookup_2043130$integer_lookup_130/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*9
_class/
-+loc:@embedding_236/embedding_lookup/2043130*+
_output_shapes
:?????????*
dtype02 
embedding_236/embedding_lookup?
'embedding_236/embedding_lookup/IdentityIdentity'embedding_236/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*9
_class/
-+loc:@embedding_236/embedding_lookup/2043130*+
_output_shapes
:?????????2)
'embedding_236/embedding_lookup/Identity?
)embedding_236/embedding_lookup/Identity_1Identity0embedding_236/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2+
)embedding_236/embedding_lookup/Identity_1u
flatten_45/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_45/Const?
flatten_45/ReshapeReshape2embedding_236/embedding_lookup/Identity_1:output:0flatten_45/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_45/Reshapeu
flatten_46/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_46/Const?
flatten_46/ReshapeReshape2embedding_237/embedding_lookup/Identity_1:output:0flatten_46/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_46/Reshapeu
flatten_47/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_47/Const?
flatten_47/ReshapeReshape2embedding_238/embedding_lookup/Identity_1:output:0flatten_47/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_47/Reshape?
normalization_348/subSubinputs_3normalization_348_sub_y*
T0*'
_output_shapes
:?????????2
normalization_348/sub{
normalization_348/SqrtSqrtnormalization_348_sqrt_x*
T0*
_output_shapes

:2
normalization_348/Sqrt
normalization_348/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_348/Maximum/y?
normalization_348/MaximumMaximumnormalization_348/Sqrt:y:0$normalization_348/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_348/Maximum?
normalization_348/truedivRealDivnormalization_348/sub:z:0normalization_348/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_348/truediv?
normalization_349/subSubinputs_4normalization_349_sub_y*
T0*'
_output_shapes
:?????????2
normalization_349/sub{
normalization_349/SqrtSqrtnormalization_349_sqrt_x*
T0*
_output_shapes

:2
normalization_349/Sqrt
normalization_349/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_349/Maximum/y?
normalization_349/MaximumMaximumnormalization_349/Sqrt:y:0$normalization_349/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_349/Maximum?
normalization_349/truedivRealDivnormalization_349/sub:z:0normalization_349/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_349/truediv?
normalization_350/subSubinputs_5normalization_350_sub_y*
T0*'
_output_shapes
:?????????2
normalization_350/sub{
normalization_350/SqrtSqrtnormalization_350_sqrt_x*
T0*
_output_shapes

:2
normalization_350/Sqrt
normalization_350/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_350/Maximum/y?
normalization_350/MaximumMaximumnormalization_350/Sqrt:y:0$normalization_350/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_350/Maximum?
normalization_350/truedivRealDivnormalization_350/sub:z:0normalization_350/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_350/truediv?
normalization_351/subSubinputs_6normalization_351_sub_y*
T0*'
_output_shapes
:?????????2
normalization_351/sub{
normalization_351/SqrtSqrtnormalization_351_sqrt_x*
T0*
_output_shapes

:2
normalization_351/Sqrt
normalization_351/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_351/Maximum/y?
normalization_351/MaximumMaximumnormalization_351/Sqrt:y:0$normalization_351/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_351/Maximum?
normalization_351/truedivRealDivnormalization_351/sub:z:0normalization_351/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_351/truediv?
normalization_352/subSubinputs_7normalization_352_sub_y*
T0*'
_output_shapes
:?????????2
normalization_352/sub{
normalization_352/SqrtSqrtnormalization_352_sqrt_x*
T0*
_output_shapes

:2
normalization_352/Sqrt
normalization_352/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_352/Maximum/y?
normalization_352/MaximumMaximumnormalization_352/Sqrt:y:0$normalization_352/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_352/Maximum?
normalization_352/truedivRealDivnormalization_352/sub:z:0normalization_352/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_352/truediv?
normalization_353/subSubinputs_8normalization_353_sub_y*
T0*'
_output_shapes
:?????????2
normalization_353/sub{
normalization_353/SqrtSqrtnormalization_353_sqrt_x*
T0*
_output_shapes

:2
normalization_353/Sqrt
normalization_353/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_353/Maximum/y?
normalization_353/MaximumMaximumnormalization_353/Sqrt:y:0$normalization_353/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_353/Maximum?
normalization_353/truedivRealDivnormalization_353/sub:z:0normalization_353/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_353/truedivz
concatenate_58/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_58/concat/axis?
concatenate_58/concatConcatV2flatten_45/Reshape:output:0flatten_46/Reshape:output:0flatten_47/Reshape:output:0normalization_348/truediv:z:0normalization_349/truediv:z:0normalization_350/truediv:z:0normalization_351/truediv:z:0normalization_352/truediv:z:0normalization_353/truediv:z:0#concatenate_58/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????)2
concatenate_58/concat?
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:)
*
dtype02 
dense_59/MatMul/ReadVariableOp?
dense_59/MatMulMatMulconcatenate_58/concat:output:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_59/MatMul?
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_59/BiasAdd/ReadVariableOp?
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_59/BiasAdds
dense_59/ReluReludense_59/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_59/Relu?
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_60/MatMul/ReadVariableOp?
dense_60/MatMulMatMuldense_59/Relu:activations:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_60/MatMul?
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_60/BiasAdd/ReadVariableOp?
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_60/BiasAddt
IdentityIdentitydense_60/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp^embedding_236/embedding_lookup^embedding_237/embedding_lookup^embedding_238/embedding_lookup7^integer_lookup_130/hash_table_Lookup/LookupTableFindV25^string_lookup_84/hash_table_Lookup/LookupTableFindV25^string_lookup_85/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : ::::::::::::: : : : 2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2@
embedding_236/embedding_lookupembedding_236/embedding_lookup2@
embedding_237/embedding_lookupembedding_237/embedding_lookup2@
embedding_238/embedding_lookupembedding_238/embedding_lookup2p
6integer_lookup_130/hash_table_Lookup/LookupTableFindV26integer_lookup_130/hash_table_Lookup/LookupTableFindV22l
4string_lookup_84/hash_table_Lookup/LookupTableFindV24string_lookup_84/hash_table_Lookup/LookupTableFindV22l
4string_lookup_85/hash_table_Lookup/LookupTableFindV24string_lookup_85/hash_table_Lookup/LookupTableFindV2:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
0__inference_concatenate_58_layer_call_fn_2043535
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????)* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_concatenate_58_layer_call_and_return_conditional_losses_20424052
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????)2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8
?

?
E__inference_dense_60_layer_call_and_return_conditional_losses_2042434

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
H
__inference__creator_2043663
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_1653829*
value_dtype0	2
MutableHashTablei
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identitya
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
?
 __inference__initializer_2043587:
6key_value_init1653681_lookuptableimportv2_table_handle2
.key_value_init1653681_lookuptableimportv2_keys	4
0key_value_init1653681_lookuptableimportv2_values	
identity??)key_value_init1653681/LookupTableImportV2?
)key_value_init1653681/LookupTableImportV2LookupTableImportV26key_value_init1653681_lookuptableimportv2_table_handle.key_value_init1653681_lookuptableimportv2_keys0key_value_init1653681_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 2+
)key_value_init1653681/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityz
NoOpNoOp*^key_value_init1653681/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init1653681/LookupTableImportV2)key_value_init1653681/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference_save_fn_2043719
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2A
?MutableHashTable_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1Q
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const\

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:2

Identity_2W

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1^

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:2

Identity_5?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
__inference_save_fn_2043746
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2A
?MutableHashTable_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1Q
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const\

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:2

Identity_2W

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1^

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:2

Identity_5?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
.
__inference__destroyer_2043607
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
,
__inference_<lambda>_2043767
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
%__inference_signature_wrapper_2043097
degree_centrality
dist_to_dhoby
dist_to_nearest_stn
eigenvector_centrality
flat_model_type
floor_area_sqm	
month	
remaining_lease_years
storey_range
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20:)


unknown_21:


unknown_22:


unknown_23:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmonthflat_model_typestorey_rangefloor_area_sqmdist_to_nearest_stndist_to_dhobydegree_centralityeigenvector_centralityremaining_lease_yearsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*-
Tin&
$2"				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	 !*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_20422512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : ::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:?????????
+
_user_specified_namedegree_centrality:VR
'
_output_shapes
:?????????
'
_user_specified_namedist_to_dhoby:\X
'
_output_shapes
:?????????
-
_user_specified_namedist_to_nearest_stn:_[
'
_output_shapes
:?????????
0
_user_specified_nameeigenvector_centrality:XT
'
_output_shapes
:?????????
)
_user_specified_nameflat_model_type:WS
'
_output_shapes
:?????????
(
_user_specified_namefloor_area_sqm:NJ
'
_output_shapes
:?????????

_user_specified_namemonth:^Z
'
_output_shapes
:?????????
/
_user_specified_nameremaining_lease_years:UQ
'
_output_shapes
:?????????
&
_user_specified_namestorey_range:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
.
__inference__destroyer_2043592
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
J__inference_embedding_237_layer_call_and_return_conditional_losses_2042308

inputs	*
embedding_lookup_2042302:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2042302inputs",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/2042302*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@embedding_lookup/2042302*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
? 
#__inference__traced_restore_2044203
file_prefixM
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable:	 Q
Gmutablehashtable_table_restore_1_lookuptableimportv2_mutablehashtable_1: Q
Gmutablehashtable_table_restore_2_lookuptableimportv2_mutablehashtable_2: ;
)assignvariableop_embedding_236_embeddings:=
+assignvariableop_1_embedding_237_embeddings:=
+assignvariableop_2_embedding_238_embeddings:%
assignvariableop_3_mean:)
assignvariableop_4_variance:"
assignvariableop_5_count:	 '
assignvariableop_6_mean_1:+
assignvariableop_7_variance_1:$
assignvariableop_8_count_1:	 '
assignvariableop_9_mean_2:,
assignvariableop_10_variance_2:%
assignvariableop_11_count_2:	 (
assignvariableop_12_mean_3:,
assignvariableop_13_variance_3:%
assignvariableop_14_count_3:	 (
assignvariableop_15_mean_4:,
assignvariableop_16_variance_4:%
assignvariableop_17_count_4:	 (
assignvariableop_18_mean_5:,
assignvariableop_19_variance_5:%
assignvariableop_20_count_5:	 5
#assignvariableop_21_dense_59_kernel:)
/
!assignvariableop_22_dense_59_bias:
5
#assignvariableop_23_dense_60_kernel:
/
!assignvariableop_24_dense_60_bias:'
assignvariableop_25_adam_iter:	 )
assignvariableop_26_adam_beta_1: )
assignvariableop_27_adam_beta_2: (
assignvariableop_28_adam_decay: 0
&assignvariableop_29_adam_learning_rate: #
assignvariableop_30_total: %
assignvariableop_31_count_6: %
assignvariableop_32_total_1: %
assignvariableop_33_count_7: %
assignvariableop_34_total_2: %
assignvariableop_35_count_8: %
assignvariableop_36_total_3: %
assignvariableop_37_count_9: E
3assignvariableop_38_adam_embedding_236_embeddings_m:E
3assignvariableop_39_adam_embedding_237_embeddings_m:E
3assignvariableop_40_adam_embedding_238_embeddings_m:<
*assignvariableop_41_adam_dense_59_kernel_m:)
6
(assignvariableop_42_adam_dense_59_bias_m:
<
*assignvariableop_43_adam_dense_60_kernel_m:
6
(assignvariableop_44_adam_dense_60_bias_m:E
3assignvariableop_45_adam_embedding_236_embeddings_v:E
3assignvariableop_46_adam_embedding_237_embeddings_v:E
3assignvariableop_47_adam_embedding_238_embeddings_v:<
*assignvariableop_48_adam_dense_59_kernel_v:)
6
(assignvariableop_49_adam_dense_59_bias_v:
<
*assignvariableop_50_adam_dense_60_kernel_v:
6
(assignvariableop_51_adam_dense_60_bias_v:
identity_53??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?4MutableHashTable_table_restore_1/LookupTableImportV2?4MutableHashTable_table_restore_2/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B?;B8layer_with_weights-0/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-0/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-1/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-1/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-2/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-2/token_counts/.ATTRIBUTES/table-valuesB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-7/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-8/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-9/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-10/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-11/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/count/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*?
value?B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*I
dtypes?
=2;											2
	RestoreV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:0RestoreV2:tensors:1*	
Tin0	*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 24
2MutableHashTable_table_restore/LookupTableImportV2?
4MutableHashTable_table_restore_1/LookupTableImportV2LookupTableImportV2Gmutablehashtable_table_restore_1_lookuptableimportv2_mutablehashtable_1RestoreV2:tensors:2RestoreV2:tensors:3*	
Tin0*

Tout0	*%
_class
loc:@MutableHashTable_1*
_output_shapes
 26
4MutableHashTable_table_restore_1/LookupTableImportV2?
4MutableHashTable_table_restore_2/LookupTableImportV2LookupTableImportV2Gmutablehashtable_table_restore_2_lookuptableimportv2_mutablehashtable_2RestoreV2:tensors:4RestoreV2:tensors:5*	
Tin0*

Tout0	*%
_class
loc:@MutableHashTable_2*
_output_shapes
 26
4MutableHashTable_table_restore_2/LookupTableImportV2g
IdentityIdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp)assignvariableop_embedding_236_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_embedding_237_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp+assignvariableop_2_embedding_238_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3l

Identity_4IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4l

Identity_5IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_countIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5l

Identity_6IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_mean_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6l

Identity_7IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_variance_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7l

Identity_8IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_count_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8l

Identity_9IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_mean_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_variance_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_mean_3Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_variance_3Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_3Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_mean_4Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_variance_4Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_4Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_mean_5Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_variance_5Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_5Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp#assignvariableop_21_dense_59_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp!assignvariableop_22_dense_59_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp#assignvariableop_23_dense_60_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp!assignvariableop_24_dense_60_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:31"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_iterIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_beta_2Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_decayIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_learning_rateIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_totalIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_count_6Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_7Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_total_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_count_8Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_total_3Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_count_9Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_embedding_236_embeddings_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp3assignvariableop_39_adam_embedding_237_embeddings_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp3assignvariableop_40_adam_embedding_238_embeddings_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_59_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_59_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_60_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_60_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp3assignvariableop_45_adam_embedding_236_embeddings_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp3assignvariableop_46_adam_embedding_237_embeddings_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp3assignvariableop_47_adam_embedding_238_embeddings_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_dense_59_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense_59_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_dense_60_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_dense_60_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_519
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV25^MutableHashTable_table_restore_1/LookupTableImportV25^MutableHashTable_table_restore_2/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_52f
Identity_53IdentityIdentity_52:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_53?

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV25^MutableHashTable_table_restore_1/LookupTableImportV25^MutableHashTable_table_restore_2/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_53Identity_53:output:0*?
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV22l
4MutableHashTable_table_restore_1/LookupTableImportV24MutableHashTable_table_restore_1/LookupTableImportV22l
4MutableHashTable_table_restore_2/LookupTableImportV24MutableHashTable_table_restore_2/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable:+'
%
_class
loc:@MutableHashTable_1:+'
%
_class
loc:@MutableHashTable_2
?,
?
__inference_adapt_step_1661549
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNexts
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
.
__inference__destroyer_2043658
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
E__inference_dense_59_layer_call_and_return_conditional_losses_2043546

inputs0
matmul_readvariableop_resource:)
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:)
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????): : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?	
?
__inference_restore_fn_2043727
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 24
2MutableHashTable_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
?
E__inference_dense_59_layer_call_and_return_conditional_losses_2042418

inputs0
matmul_readvariableop_resource:)
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:)
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????): : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????)
 
_user_specified_nameinputs
?
?
K__inference_concatenate_58_layer_call_and_return_conditional_losses_2043522
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????)2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????)2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8
?
0
 __inference__initializer_2043602
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
c
G__inference_flatten_45_layer_call_and_return_conditional_losses_2043481

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
0
 __inference__initializer_2043668
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference_adapt_step_1661313
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNextq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shape?
ReshapeReshapeIteratorGetNext:components:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshape?
UniqueWithCountsUniqueWithCountsReshape:output:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	2
UniqueWithCounts?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:2*
(None_lookup_table_find/LookupTableFindV2?
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2
add?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2.
,None_lookup_table_insert/LookupTableInsertV2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?
c
G__inference_flatten_46_layer_call_and_return_conditional_losses_2043492

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
Ε
?
"__inference__wrapped_model_2042251	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_yearsN
Jmodel_29_string_lookup_85_hash_table_lookup_lookuptablefindv2_table_handleO
Kmodel_29_string_lookup_85_hash_table_lookup_lookuptablefindv2_default_value	N
Jmodel_29_string_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleO
Kmodel_29_string_lookup_84_hash_table_lookup_lookuptablefindv2_default_value	P
Lmodel_29_integer_lookup_130_hash_table_lookup_lookuptablefindv2_table_handleQ
Mmodel_29_integer_lookup_130_hash_table_lookup_lookuptablefindv2_default_value	A
/model_29_embedding_238_embedding_lookup_2042172:A
/model_29_embedding_237_embedding_lookup_2042177:A
/model_29_embedding_236_embedding_lookup_2042182:$
 model_29_normalization_348_sub_y%
!model_29_normalization_348_sqrt_x$
 model_29_normalization_349_sub_y%
!model_29_normalization_349_sqrt_x$
 model_29_normalization_350_sub_y%
!model_29_normalization_350_sqrt_x$
 model_29_normalization_351_sub_y%
!model_29_normalization_351_sqrt_x$
 model_29_normalization_352_sub_y%
!model_29_normalization_352_sqrt_x$
 model_29_normalization_353_sub_y%
!model_29_normalization_353_sqrt_xB
0model_29_dense_59_matmul_readvariableop_resource:)
?
1model_29_dense_59_biasadd_readvariableop_resource:
B
0model_29_dense_60_matmul_readvariableop_resource:
?
1model_29_dense_60_biasadd_readvariableop_resource:
identity??(model_29/dense_59/BiasAdd/ReadVariableOp?'model_29/dense_59/MatMul/ReadVariableOp?(model_29/dense_60/BiasAdd/ReadVariableOp?'model_29/dense_60/MatMul/ReadVariableOp?'model_29/embedding_236/embedding_lookup?'model_29/embedding_237/embedding_lookup?'model_29/embedding_238/embedding_lookup??model_29/integer_lookup_130/hash_table_Lookup/LookupTableFindV2?=model_29/string_lookup_84/hash_table_Lookup/LookupTableFindV2?=model_29/string_lookup_85/hash_table_Lookup/LookupTableFindV2?
=model_29/string_lookup_85/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Jmodel_29_string_lookup_85_hash_table_lookup_lookuptablefindv2_table_handlestorey_rangeKmodel_29_string_lookup_85_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2?
=model_29/string_lookup_85/hash_table_Lookup/LookupTableFindV2?
"model_29/string_lookup_85/IdentityIdentityFmodel_29/string_lookup_85/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2$
"model_29/string_lookup_85/Identity?
=model_29/string_lookup_84/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Jmodel_29_string_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleflat_model_typeKmodel_29_string_lookup_84_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2?
=model_29/string_lookup_84/hash_table_Lookup/LookupTableFindV2?
"model_29/string_lookup_84/IdentityIdentityFmodel_29/string_lookup_84/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2$
"model_29/string_lookup_84/Identity?
?model_29/integer_lookup_130/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Lmodel_29_integer_lookup_130_hash_table_lookup_lookuptablefindv2_table_handlemonthMmodel_29_integer_lookup_130_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????2A
?model_29/integer_lookup_130/hash_table_Lookup/LookupTableFindV2?
$model_29/integer_lookup_130/IdentityIdentityHmodel_29/integer_lookup_130/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2&
$model_29/integer_lookup_130/Identity?
'model_29/embedding_238/embedding_lookupResourceGather/model_29_embedding_238_embedding_lookup_2042172+model_29/string_lookup_85/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*B
_class8
64loc:@model_29/embedding_238/embedding_lookup/2042172*+
_output_shapes
:?????????*
dtype02)
'model_29/embedding_238/embedding_lookup?
0model_29/embedding_238/embedding_lookup/IdentityIdentity0model_29/embedding_238/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@model_29/embedding_238/embedding_lookup/2042172*+
_output_shapes
:?????????22
0model_29/embedding_238/embedding_lookup/Identity?
2model_29/embedding_238/embedding_lookup/Identity_1Identity9model_29/embedding_238/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????24
2model_29/embedding_238/embedding_lookup/Identity_1?
'model_29/embedding_237/embedding_lookupResourceGather/model_29_embedding_237_embedding_lookup_2042177+model_29/string_lookup_84/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*B
_class8
64loc:@model_29/embedding_237/embedding_lookup/2042177*+
_output_shapes
:?????????*
dtype02)
'model_29/embedding_237/embedding_lookup?
0model_29/embedding_237/embedding_lookup/IdentityIdentity0model_29/embedding_237/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@model_29/embedding_237/embedding_lookup/2042177*+
_output_shapes
:?????????22
0model_29/embedding_237/embedding_lookup/Identity?
2model_29/embedding_237/embedding_lookup/Identity_1Identity9model_29/embedding_237/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????24
2model_29/embedding_237/embedding_lookup/Identity_1?
'model_29/embedding_236/embedding_lookupResourceGather/model_29_embedding_236_embedding_lookup_2042182-model_29/integer_lookup_130/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*B
_class8
64loc:@model_29/embedding_236/embedding_lookup/2042182*+
_output_shapes
:?????????*
dtype02)
'model_29/embedding_236/embedding_lookup?
0model_29/embedding_236/embedding_lookup/IdentityIdentity0model_29/embedding_236/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@model_29/embedding_236/embedding_lookup/2042182*+
_output_shapes
:?????????22
0model_29/embedding_236/embedding_lookup/Identity?
2model_29/embedding_236/embedding_lookup/Identity_1Identity9model_29/embedding_236/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????24
2model_29/embedding_236/embedding_lookup/Identity_1?
model_29/flatten_45/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_29/flatten_45/Const?
model_29/flatten_45/ReshapeReshape;model_29/embedding_236/embedding_lookup/Identity_1:output:0"model_29/flatten_45/Const:output:0*
T0*'
_output_shapes
:?????????2
model_29/flatten_45/Reshape?
model_29/flatten_46/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_29/flatten_46/Const?
model_29/flatten_46/ReshapeReshape;model_29/embedding_237/embedding_lookup/Identity_1:output:0"model_29/flatten_46/Const:output:0*
T0*'
_output_shapes
:?????????2
model_29/flatten_46/Reshape?
model_29/flatten_47/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_29/flatten_47/Const?
model_29/flatten_47/ReshapeReshape;model_29/embedding_238/embedding_lookup/Identity_1:output:0"model_29/flatten_47/Const:output:0*
T0*'
_output_shapes
:?????????2
model_29/flatten_47/Reshape?
model_29/normalization_348/subSubfloor_area_sqm model_29_normalization_348_sub_y*
T0*'
_output_shapes
:?????????2 
model_29/normalization_348/sub?
model_29/normalization_348/SqrtSqrt!model_29_normalization_348_sqrt_x*
T0*
_output_shapes

:2!
model_29/normalization_348/Sqrt?
$model_29/normalization_348/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32&
$model_29/normalization_348/Maximum/y?
"model_29/normalization_348/MaximumMaximum#model_29/normalization_348/Sqrt:y:0-model_29/normalization_348/Maximum/y:output:0*
T0*
_output_shapes

:2$
"model_29/normalization_348/Maximum?
"model_29/normalization_348/truedivRealDiv"model_29/normalization_348/sub:z:0&model_29/normalization_348/Maximum:z:0*
T0*'
_output_shapes
:?????????2$
"model_29/normalization_348/truediv?
model_29/normalization_349/subSubdist_to_nearest_stn model_29_normalization_349_sub_y*
T0*'
_output_shapes
:?????????2 
model_29/normalization_349/sub?
model_29/normalization_349/SqrtSqrt!model_29_normalization_349_sqrt_x*
T0*
_output_shapes

:2!
model_29/normalization_349/Sqrt?
$model_29/normalization_349/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32&
$model_29/normalization_349/Maximum/y?
"model_29/normalization_349/MaximumMaximum#model_29/normalization_349/Sqrt:y:0-model_29/normalization_349/Maximum/y:output:0*
T0*
_output_shapes

:2$
"model_29/normalization_349/Maximum?
"model_29/normalization_349/truedivRealDiv"model_29/normalization_349/sub:z:0&model_29/normalization_349/Maximum:z:0*
T0*'
_output_shapes
:?????????2$
"model_29/normalization_349/truediv?
model_29/normalization_350/subSubdist_to_dhoby model_29_normalization_350_sub_y*
T0*'
_output_shapes
:?????????2 
model_29/normalization_350/sub?
model_29/normalization_350/SqrtSqrt!model_29_normalization_350_sqrt_x*
T0*
_output_shapes

:2!
model_29/normalization_350/Sqrt?
$model_29/normalization_350/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32&
$model_29/normalization_350/Maximum/y?
"model_29/normalization_350/MaximumMaximum#model_29/normalization_350/Sqrt:y:0-model_29/normalization_350/Maximum/y:output:0*
T0*
_output_shapes

:2$
"model_29/normalization_350/Maximum?
"model_29/normalization_350/truedivRealDiv"model_29/normalization_350/sub:z:0&model_29/normalization_350/Maximum:z:0*
T0*'
_output_shapes
:?????????2$
"model_29/normalization_350/truediv?
model_29/normalization_351/subSubdegree_centrality model_29_normalization_351_sub_y*
T0*'
_output_shapes
:?????????2 
model_29/normalization_351/sub?
model_29/normalization_351/SqrtSqrt!model_29_normalization_351_sqrt_x*
T0*
_output_shapes

:2!
model_29/normalization_351/Sqrt?
$model_29/normalization_351/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32&
$model_29/normalization_351/Maximum/y?
"model_29/normalization_351/MaximumMaximum#model_29/normalization_351/Sqrt:y:0-model_29/normalization_351/Maximum/y:output:0*
T0*
_output_shapes

:2$
"model_29/normalization_351/Maximum?
"model_29/normalization_351/truedivRealDiv"model_29/normalization_351/sub:z:0&model_29/normalization_351/Maximum:z:0*
T0*'
_output_shapes
:?????????2$
"model_29/normalization_351/truediv?
model_29/normalization_352/subSubeigenvector_centrality model_29_normalization_352_sub_y*
T0*'
_output_shapes
:?????????2 
model_29/normalization_352/sub?
model_29/normalization_352/SqrtSqrt!model_29_normalization_352_sqrt_x*
T0*
_output_shapes

:2!
model_29/normalization_352/Sqrt?
$model_29/normalization_352/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32&
$model_29/normalization_352/Maximum/y?
"model_29/normalization_352/MaximumMaximum#model_29/normalization_352/Sqrt:y:0-model_29/normalization_352/Maximum/y:output:0*
T0*
_output_shapes

:2$
"model_29/normalization_352/Maximum?
"model_29/normalization_352/truedivRealDiv"model_29/normalization_352/sub:z:0&model_29/normalization_352/Maximum:z:0*
T0*'
_output_shapes
:?????????2$
"model_29/normalization_352/truediv?
model_29/normalization_353/subSubremaining_lease_years model_29_normalization_353_sub_y*
T0*'
_output_shapes
:?????????2 
model_29/normalization_353/sub?
model_29/normalization_353/SqrtSqrt!model_29_normalization_353_sqrt_x*
T0*
_output_shapes

:2!
model_29/normalization_353/Sqrt?
$model_29/normalization_353/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32&
$model_29/normalization_353/Maximum/y?
"model_29/normalization_353/MaximumMaximum#model_29/normalization_353/Sqrt:y:0-model_29/normalization_353/Maximum/y:output:0*
T0*
_output_shapes

:2$
"model_29/normalization_353/Maximum?
"model_29/normalization_353/truedivRealDiv"model_29/normalization_353/sub:z:0&model_29/normalization_353/Maximum:z:0*
T0*'
_output_shapes
:?????????2$
"model_29/normalization_353/truediv?
#model_29/concatenate_58/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_29/concatenate_58/concat/axis?
model_29/concatenate_58/concatConcatV2$model_29/flatten_45/Reshape:output:0$model_29/flatten_46/Reshape:output:0$model_29/flatten_47/Reshape:output:0&model_29/normalization_348/truediv:z:0&model_29/normalization_349/truediv:z:0&model_29/normalization_350/truediv:z:0&model_29/normalization_351/truediv:z:0&model_29/normalization_352/truediv:z:0&model_29/normalization_353/truediv:z:0,model_29/concatenate_58/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????)2 
model_29/concatenate_58/concat?
'model_29/dense_59/MatMul/ReadVariableOpReadVariableOp0model_29_dense_59_matmul_readvariableop_resource*
_output_shapes

:)
*
dtype02)
'model_29/dense_59/MatMul/ReadVariableOp?
model_29/dense_59/MatMulMatMul'model_29/concatenate_58/concat:output:0/model_29/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model_29/dense_59/MatMul?
(model_29/dense_59/BiasAdd/ReadVariableOpReadVariableOp1model_29_dense_59_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(model_29/dense_59/BiasAdd/ReadVariableOp?
model_29/dense_59/BiasAddBiasAdd"model_29/dense_59/MatMul:product:00model_29/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model_29/dense_59/BiasAdd?
model_29/dense_59/ReluRelu"model_29/dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
model_29/dense_59/Relu?
'model_29/dense_60/MatMul/ReadVariableOpReadVariableOp0model_29_dense_60_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'model_29/dense_60/MatMul/ReadVariableOp?
model_29/dense_60/MatMulMatMul$model_29/dense_59/Relu:activations:0/model_29/dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_29/dense_60/MatMul?
(model_29/dense_60/BiasAdd/ReadVariableOpReadVariableOp1model_29_dense_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_29/dense_60/BiasAdd/ReadVariableOp?
model_29/dense_60/BiasAddBiasAdd"model_29/dense_60/MatMul:product:00model_29/dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_29/dense_60/BiasAdd}
IdentityIdentity"model_29/dense_60/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp)^model_29/dense_59/BiasAdd/ReadVariableOp(^model_29/dense_59/MatMul/ReadVariableOp)^model_29/dense_60/BiasAdd/ReadVariableOp(^model_29/dense_60/MatMul/ReadVariableOp(^model_29/embedding_236/embedding_lookup(^model_29/embedding_237/embedding_lookup(^model_29/embedding_238/embedding_lookup@^model_29/integer_lookup_130/hash_table_Lookup/LookupTableFindV2>^model_29/string_lookup_84/hash_table_Lookup/LookupTableFindV2>^model_29/string_lookup_85/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : ::::::::::::: : : : 2T
(model_29/dense_59/BiasAdd/ReadVariableOp(model_29/dense_59/BiasAdd/ReadVariableOp2R
'model_29/dense_59/MatMul/ReadVariableOp'model_29/dense_59/MatMul/ReadVariableOp2T
(model_29/dense_60/BiasAdd/ReadVariableOp(model_29/dense_60/BiasAdd/ReadVariableOp2R
'model_29/dense_60/MatMul/ReadVariableOp'model_29/dense_60/MatMul/ReadVariableOp2R
'model_29/embedding_236/embedding_lookup'model_29/embedding_236/embedding_lookup2R
'model_29/embedding_237/embedding_lookup'model_29/embedding_237/embedding_lookup2R
'model_29/embedding_238/embedding_lookup'model_29/embedding_238/embedding_lookup2?
?model_29/integer_lookup_130/hash_table_Lookup/LookupTableFindV2?model_29/integer_lookup_130/hash_table_Lookup/LookupTableFindV22~
=model_29/string_lookup_84/hash_table_Lookup/LookupTableFindV2=model_29/string_lookup_84/hash_table_Lookup/LookupTableFindV22~
=model_29/string_lookup_85/hash_table_Lookup/LookupTableFindV2=model_29/string_lookup_85/hash_table_Lookup/LookupTableFindV2:N J
'
_output_shapes
:?????????

_user_specified_namemonth:XT
'
_output_shapes
:?????????
)
_user_specified_nameflat_model_type:UQ
'
_output_shapes
:?????????
&
_user_specified_namestorey_range:WS
'
_output_shapes
:?????????
(
_user_specified_namefloor_area_sqm:\X
'
_output_shapes
:?????????
-
_user_specified_namedist_to_nearest_stn:VR
'
_output_shapes
:?????????
'
_user_specified_namedist_to_dhoby:ZV
'
_output_shapes
:?????????
+
_user_specified_namedegree_centrality:_[
'
_output_shapes
:?????????
0
_user_specified_nameeigenvector_centrality:^Z
'
_output_shapes
:?????????
/
_user_specified_nameremaining_lease_years:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
K__inference_concatenate_58_layer_call_and_return_conditional_losses_2042405

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????)2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????)2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_adapt_step_1661327
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNextq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shape?
ReshapeReshapeIteratorGetNext:components:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshape?
UniqueWithCountsUniqueWithCountsReshape:output:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	2
UniqueWithCounts?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:2*
(None_lookup_table_find/LookupTableFindV2?
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2
add?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2.
,None_lookup_table_insert/LookupTableInsertV2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?s
?
E__inference_model_29_layer_call_and_return_conditional_losses_2042732

inputs	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8E
Astring_lookup_85_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_85_hash_table_lookup_lookuptablefindv2_default_value	E
Astring_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_84_hash_table_lookup_lookuptablefindv2_default_value	G
Cinteger_lookup_130_hash_table_lookup_lookuptablefindv2_table_handleH
Dinteger_lookup_130_hash_table_lookup_lookuptablefindv2_default_value	'
embedding_238_2042666:'
embedding_237_2042669:'
embedding_236_2042672:
normalization_348_sub_y
normalization_348_sqrt_x
normalization_349_sub_y
normalization_349_sqrt_x
normalization_350_sub_y
normalization_350_sqrt_x
normalization_351_sub_y
normalization_351_sqrt_x
normalization_352_sub_y
normalization_352_sqrt_x
normalization_353_sub_y
normalization_353_sqrt_x"
dense_59_2042721:)

dense_59_2042723:
"
dense_60_2042726:

dense_60_2042728:
identity?? dense_59/StatefulPartitionedCall? dense_60/StatefulPartitionedCall?%embedding_236/StatefulPartitionedCall?%embedding_237/StatefulPartitionedCall?%embedding_238/StatefulPartitionedCall?6integer_lookup_130/hash_table_Lookup/LookupTableFindV2?4string_lookup_84/hash_table_Lookup/LookupTableFindV2?4string_lookup_85/hash_table_Lookup/LookupTableFindV2?
4string_lookup_85/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_85_hash_table_lookup_lookuptablefindv2_table_handleinputs_2Bstring_lookup_85_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_85/hash_table_Lookup/LookupTableFindV2?
string_lookup_85/IdentityIdentity=string_lookup_85/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_85/Identity?
4string_lookup_84/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_84_hash_table_lookup_lookuptablefindv2_table_handleinputs_1Bstring_lookup_84_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_84/hash_table_Lookup/LookupTableFindV2?
string_lookup_84/IdentityIdentity=string_lookup_84/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_84/Identity?
6integer_lookup_130/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Cinteger_lookup_130_hash_table_lookup_lookuptablefindv2_table_handleinputsDinteger_lookup_130_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????28
6integer_lookup_130/hash_table_Lookup/LookupTableFindV2?
integer_lookup_130/IdentityIdentity?integer_lookup_130/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup_130/Identity?
%embedding_238/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_85/Identity:output:0embedding_238_2042666*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_238_layer_call_and_return_conditional_losses_20422952'
%embedding_238/StatefulPartitionedCall?
%embedding_237/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_84/Identity:output:0embedding_237_2042669*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_237_layer_call_and_return_conditional_losses_20423082'
%embedding_237/StatefulPartitionedCall?
%embedding_236/StatefulPartitionedCallStatefulPartitionedCall$integer_lookup_130/Identity:output:0embedding_236_2042672*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_236_layer_call_and_return_conditional_losses_20423212'
%embedding_236/StatefulPartitionedCall?
flatten_45/PartitionedCallPartitionedCall.embedding_236/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_45_layer_call_and_return_conditional_losses_20423312
flatten_45/PartitionedCall?
flatten_46/PartitionedCallPartitionedCall.embedding_237/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_46_layer_call_and_return_conditional_losses_20423392
flatten_46/PartitionedCall?
flatten_47/PartitionedCallPartitionedCall.embedding_238/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_47_layer_call_and_return_conditional_losses_20423472
flatten_47/PartitionedCall?
normalization_348/subSubinputs_3normalization_348_sub_y*
T0*'
_output_shapes
:?????????2
normalization_348/sub{
normalization_348/SqrtSqrtnormalization_348_sqrt_x*
T0*
_output_shapes

:2
normalization_348/Sqrt
normalization_348/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_348/Maximum/y?
normalization_348/MaximumMaximumnormalization_348/Sqrt:y:0$normalization_348/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_348/Maximum?
normalization_348/truedivRealDivnormalization_348/sub:z:0normalization_348/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_348/truediv?
normalization_349/subSubinputs_4normalization_349_sub_y*
T0*'
_output_shapes
:?????????2
normalization_349/sub{
normalization_349/SqrtSqrtnormalization_349_sqrt_x*
T0*
_output_shapes

:2
normalization_349/Sqrt
normalization_349/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_349/Maximum/y?
normalization_349/MaximumMaximumnormalization_349/Sqrt:y:0$normalization_349/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_349/Maximum?
normalization_349/truedivRealDivnormalization_349/sub:z:0normalization_349/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_349/truediv?
normalization_350/subSubinputs_5normalization_350_sub_y*
T0*'
_output_shapes
:?????????2
normalization_350/sub{
normalization_350/SqrtSqrtnormalization_350_sqrt_x*
T0*
_output_shapes

:2
normalization_350/Sqrt
normalization_350/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_350/Maximum/y?
normalization_350/MaximumMaximumnormalization_350/Sqrt:y:0$normalization_350/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_350/Maximum?
normalization_350/truedivRealDivnormalization_350/sub:z:0normalization_350/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_350/truediv?
normalization_351/subSubinputs_6normalization_351_sub_y*
T0*'
_output_shapes
:?????????2
normalization_351/sub{
normalization_351/SqrtSqrtnormalization_351_sqrt_x*
T0*
_output_shapes

:2
normalization_351/Sqrt
normalization_351/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_351/Maximum/y?
normalization_351/MaximumMaximumnormalization_351/Sqrt:y:0$normalization_351/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_351/Maximum?
normalization_351/truedivRealDivnormalization_351/sub:z:0normalization_351/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_351/truediv?
normalization_352/subSubinputs_7normalization_352_sub_y*
T0*'
_output_shapes
:?????????2
normalization_352/sub{
normalization_352/SqrtSqrtnormalization_352_sqrt_x*
T0*
_output_shapes

:2
normalization_352/Sqrt
normalization_352/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_352/Maximum/y?
normalization_352/MaximumMaximumnormalization_352/Sqrt:y:0$normalization_352/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_352/Maximum?
normalization_352/truedivRealDivnormalization_352/sub:z:0normalization_352/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_352/truediv?
normalization_353/subSubinputs_8normalization_353_sub_y*
T0*'
_output_shapes
:?????????2
normalization_353/sub{
normalization_353/SqrtSqrtnormalization_353_sqrt_x*
T0*
_output_shapes

:2
normalization_353/Sqrt
normalization_353/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_353/Maximum/y?
normalization_353/MaximumMaximumnormalization_353/Sqrt:y:0$normalization_353/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_353/Maximum?
normalization_353/truedivRealDivnormalization_353/sub:z:0normalization_353/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_353/truediv?
concatenate_58/PartitionedCallPartitionedCall#flatten_45/PartitionedCall:output:0#flatten_46/PartitionedCall:output:0#flatten_47/PartitionedCall:output:0normalization_348/truediv:z:0normalization_349/truediv:z:0normalization_350/truediv:z:0normalization_351/truediv:z:0normalization_352/truediv:z:0normalization_353/truediv:z:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????)* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_concatenate_58_layer_call_and_return_conditional_losses_20424052 
concatenate_58/PartitionedCall?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall'concatenate_58/PartitionedCall:output:0dense_59_2042721dense_59_2042723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_20424182"
 dense_59/StatefulPartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall)dense_59/StatefulPartitionedCall:output:0dense_60_2042726dense_60_2042728*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_60_layer_call_and_return_conditional_losses_20424342"
 dense_60/StatefulPartitionedCall?
IdentityIdentity)dense_60/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_59/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall&^embedding_236/StatefulPartitionedCall&^embedding_237/StatefulPartitionedCall&^embedding_238/StatefulPartitionedCall7^integer_lookup_130/hash_table_Lookup/LookupTableFindV25^string_lookup_84/hash_table_Lookup/LookupTableFindV25^string_lookup_85/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : ::::::::::::: : : : 2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2N
%embedding_236/StatefulPartitionedCall%embedding_236/StatefulPartitionedCall2N
%embedding_237/StatefulPartitionedCall%embedding_237/StatefulPartitionedCall2N
%embedding_238/StatefulPartitionedCall%embedding_238/StatefulPartitionedCall2p
6integer_lookup_130/hash_table_Lookup/LookupTableFindV26integer_lookup_130/hash_table_Lookup/LookupTableFindV22l
4string_lookup_84/hash_table_Lookup/LookupTableFindV24string_lookup_84/hash_table_Lookup/LookupTableFindV22l
4string_lookup_85/hash_table_Lookup/LookupTableFindV24string_lookup_85/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?

?
E__inference_dense_60_layer_call_and_return_conditional_losses_2043565

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
/__inference_embedding_238_layer_call_fn_2043475

inputs	
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_embedding_238_layer_call_and_return_conditional_losses_20422952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
.
__inference__destroyer_2043625
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?	
?
__inference_restore_fn_2043700
restored_tensors_0	
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0	*

Tout0	*
_output_shapes
 24
2MutableHashTable_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
<
__inference__creator_2043579
identity??
hash_table}

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name	1653682*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference_<lambda>_2043762:
6key_value_init1653681_lookuptableimportv2_table_handle2
.key_value_init1653681_lookuptableimportv2_keys	4
0key_value_init1653681_lookuptableimportv2_values	
identity??)key_value_init1653681/LookupTableImportV2?
)key_value_init1653681/LookupTableImportV2LookupTableImportV26key_value_init1653681_lookuptableimportv2_table_handle.key_value_init1653681_lookuptableimportv2_keys0key_value_init1653681_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 2+
)key_value_init1653681/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityz
NoOpNoOp*^key_value_init1653681/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init1653681/LookupTableImportV2)key_value_init1653681/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:"?L
saver_filename:0StatefulPartitionedCall_4:0StatefulPartitionedCall_58"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
degree_centrality:
#serving_default_degree_centrality:0?????????
G
dist_to_dhoby6
serving_default_dist_to_dhoby:0?????????
S
dist_to_nearest_stn<
%serving_default_dist_to_nearest_stn:0?????????
Y
eigenvector_centrality?
(serving_default_eigenvector_centrality:0?????????
K
flat_model_type8
!serving_default_flat_model_type:0?????????
I
floor_area_sqm7
 serving_default_floor_area_sqm:0?????????
7
month.
serving_default_month:0	?????????
W
remaining_lease_years>
'serving_default_remaining_lease_years:0?????????
E
storey_range5
serving_default_storey_range:0?????????>
dense_602
StatefulPartitionedCall_3:0?????????tensorflow/serving/predict:׹
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer_with_weights-6
layer-18
layer_with_weights-7
layer-19
layer_with_weights-8
layer-20
layer_with_weights-9
layer-21
layer_with_weights-10
layer-22
layer_with_weights-11
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
	optimizer
regularization_losses
	variables
trainable_variables
 	keras_api
!
signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
b
"lookup_table
#token_counts
$	keras_api
?_adapt_function"
_tf_keras_layer
b
%lookup_table
&token_counts
'	keras_api
?_adapt_function"
_tf_keras_layer
b
(lookup_table
)token_counts
*	keras_api
?_adapt_function"
_tf_keras_layer
?
+
embeddings
,regularization_losses
-	variables
.trainable_variables
/	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
0
embeddings
1regularization_losses
2	variables
3trainable_variables
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
5
embeddings
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
:regularization_losses
;	variables
<trainable_variables
=	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
>regularization_losses
?	variables
@trainable_variables
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
F
_keep_axis
G_reduce_axis
H_reduce_axis_mask
I_broadcast_shape
Jmean
J
adapt_mean
Kvariance
Kadapt_variance
	Lcount
M	keras_api
?_adapt_function"
_tf_keras_layer
?
N
_keep_axis
O_reduce_axis
P_reduce_axis_mask
Q_broadcast_shape
Rmean
R
adapt_mean
Svariance
Sadapt_variance
	Tcount
U	keras_api
?_adapt_function"
_tf_keras_layer
?
V
_keep_axis
W_reduce_axis
X_reduce_axis_mask
Y_broadcast_shape
Zmean
Z
adapt_mean
[variance
[adapt_variance
	\count
]	keras_api
?_adapt_function"
_tf_keras_layer
?
^
_keep_axis
__reduce_axis
`_reduce_axis_mask
a_broadcast_shape
bmean
b
adapt_mean
cvariance
cadapt_variance
	dcount
e	keras_api
?_adapt_function"
_tf_keras_layer
?
f
_keep_axis
g_reduce_axis
h_reduce_axis_mask
i_broadcast_shape
jmean
j
adapt_mean
kvariance
kadapt_variance
	lcount
m	keras_api
?_adapt_function"
_tf_keras_layer
?
n
_keep_axis
o_reduce_axis
p_reduce_axis_mask
q_broadcast_shape
rmean
r
adapt_mean
svariance
sadapt_variance
	tcount
u	keras_api
?_adapt_function"
_tf_keras_layer
?
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

zkernel
{bias
|regularization_losses
}	variables
~trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate+m?0m?5m?zm?{m?	?m?	?m?+v?0v?5v?zv?{v?	?v?	?v?"
	optimizer
 "
trackable_list_wrapper
?
+3
04
55
J6
K7
L8
R9
S10
T11
Z12
[13
\14
b15
c16
d17
j18
k19
l20
r21
s22
t23
z24
{25
?26
?27"
trackable_list_wrapper
S
+0
01
52
z3
{4
?5
?6"
trackable_list_wrapper
?
?layer_metrics
?layers
regularization_losses
?non_trainable_variables
?metrics
	variables
 ?layer_regularization_losses
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
*:(2embedding_236/embeddings
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
'
+0"
trackable_list_wrapper
?
?layer_metrics
?layers
,regularization_losses
?non_trainable_variables
?metrics
-	variables
 ?layer_regularization_losses
.trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2embedding_237/embeddings
 "
trackable_list_wrapper
'
00"
trackable_list_wrapper
'
00"
trackable_list_wrapper
?
?layer_metrics
?layers
1regularization_losses
?non_trainable_variables
?metrics
2	variables
 ?layer_regularization_losses
3trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2embedding_238/embeddings
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
'
50"
trackable_list_wrapper
?
?layer_metrics
?layers
6regularization_losses
?non_trainable_variables
?metrics
7	variables
 ?layer_regularization_losses
8trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
:regularization_losses
?non_trainable_variables
?metrics
;	variables
 ?layer_regularization_losses
<trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
>regularization_losses
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
@trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
Bregularization_losses
?non_trainable_variables
?metrics
C	variables
 ?layer_regularization_losses
Dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
vregularization_losses
?non_trainable_variables
?metrics
w	variables
 ?layer_regularization_losses
xtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:)
2dense_59/kernel
:
2dense_59/bias
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
?
?layer_metrics
?layers
|regularization_losses
?non_trainable_variables
?metrics
}	variables
 ?layer_regularization_losses
~trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_60/kernel
:2dense_60/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layer_metrics
?layers
?regularization_losses
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26"
trackable_list_wrapper
?
J3
K4
L5
R6
S7
T8
Z9
[10
\11
b12
c13
d14
j15
k16
l17
r18
s19
t20"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-2Adam/embedding_236/embeddings/m
/:-2Adam/embedding_237/embeddings/m
/:-2Adam/embedding_238/embeddings/m
&:$)
2Adam/dense_59/kernel/m
 :
2Adam/dense_59/bias/m
&:$
2Adam/dense_60/kernel/m
 :2Adam/dense_60/bias/m
/:-2Adam/embedding_236/embeddings/v
/:-2Adam/embedding_237/embeddings/v
/:-2Adam/embedding_238/embeddings/v
&:$)
2Adam/dense_59/kernel/v
 :
2Adam/dense_59/bias/v
&:$
2Adam/dense_60/kernel/v
 :2Adam/dense_60/bias/v
?2?
E__inference_model_29_layer_call_and_return_conditional_losses_2043199
E__inference_model_29_layer_call_and_return_conditional_losses_2043301
E__inference_model_29_layer_call_and_return_conditional_losses_2042937
E__inference_model_29_layer_call_and_return_conditional_losses_2043026?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_2042251monthflat_model_typestorey_rangefloor_area_sqmdist_to_nearest_stndist_to_dhobydegree_centralityeigenvector_centralityremaining_lease_years	"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_model_29_layer_call_fn_2042494
*__inference_model_29_layer_call_fn_2043364
*__inference_model_29_layer_call_fn_2043427
*__inference_model_29_layer_call_fn_2042848?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_adapt_step_1661299?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_1661313?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_1661327?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_236_layer_call_and_return_conditional_losses_2043436?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_236_layer_call_fn_2043443?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_237_layer_call_and_return_conditional_losses_2043452?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_237_layer_call_fn_2043459?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_238_layer_call_and_return_conditional_losses_2043468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_238_layer_call_fn_2043475?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_45_layer_call_and_return_conditional_losses_2043481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_45_layer_call_fn_2043486?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_46_layer_call_and_return_conditional_losses_2043492?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_46_layer_call_fn_2043497?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_47_layer_call_and_return_conditional_losses_2043503?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_47_layer_call_fn_2043508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_1661455?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_1661502?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_1661549?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_1661596?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_1661643?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_1661690?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_concatenate_58_layer_call_and_return_conditional_losses_2043522?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_concatenate_58_layer_call_fn_2043535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_59_layer_call_and_return_conditional_losses_2043546?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_59_layer_call_fn_2043555?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_60_layer_call_and_return_conditional_losses_2043565?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_60_layer_call_fn_2043574?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_2043097degree_centralitydist_to_dhobydist_to_nearest_stneigenvector_centralityflat_model_typefloor_area_sqmmonthremaining_lease_yearsstorey_range"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_2043579?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
 __inference__initializer_2043587?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_2043592?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_2043597?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
 __inference__initializer_2043602?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_2043607?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_2043692checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_2043700restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?	
	?	
?2?
__inference__creator_2043612?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
 __inference__initializer_2043620?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_2043625?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_2043630?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
 __inference__initializer_2043635?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_2043640?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_2043719checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_2043727restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
?2?
__inference__creator_2043645?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
 __inference__initializer_2043653?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_2043658?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_2043663?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
 __inference__initializer_2043668?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_2043673?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_2043746checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_2043754restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8
J	
Const_9
J

Const_10
J

Const_11
J

Const_12
J

Const_13
J

Const_14
J

Const_15
J

Const_16
J

Const_17
J

Const_18
J

Const_19
J

Const_20
J

Const_21
J

Const_22
J

Const_238
__inference__creator_2043579?

? 
? "? 8
__inference__creator_2043597?

? 
? "? 8
__inference__creator_2043612?

? 
? "? 8
__inference__creator_2043630?

? 
? "? 8
__inference__creator_2043645?

? 
? "? 8
__inference__creator_2043663?

? 
? "? :
__inference__destroyer_2043592?

? 
? "? :
__inference__destroyer_2043607?

? 
? "? :
__inference__destroyer_2043625?

? 
? "? :
__inference__destroyer_2043640?

? 
? "? :
__inference__destroyer_2043658?

? 
? "? :
__inference__destroyer_2043673?

? 
? "? C
 __inference__initializer_2043587"???

? 
? "? <
 __inference__initializer_2043602?

? 
? "? C
 __inference__initializer_2043620%???

? 
? "? <
 __inference__initializer_2043635?

? 
? "? C
 __inference__initializer_2043653(???

? 
? "? <
 __inference__initializer_2043668?

? 
? "? ?
"__inference__wrapped_model_2042251?*(?%?"?50+????????????z{?????
???
???
?
month?????????	
)?&
flat_model_type?????????
&?#
storey_range?????????
(?%
floor_area_sqm?????????
-?*
dist_to_nearest_stn?????????
'?$
dist_to_dhoby?????????
+?(
degree_centrality?????????
0?-
eigenvector_centrality?????????
/?,
remaining_lease_years?????????
? "3?0
.
dense_60"?
dense_60?????????n
__inference_adapt_step_1661299L#?A?>
7?4
2?/?
??????????	IteratorSpec
? "
 n
__inference_adapt_step_1661313L&?A?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_1661327L)?A?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_1661455LLJKA?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_1661502LTRSA?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_1661549L\Z[A?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_1661596LdbcA?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_1661643LljkA?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_1661690LtrsA?>
7?4
2?/?
??????????IteratorSpec
? "
 ?
K__inference_concatenate_58_layer_call_and_return_conditional_losses_2043522????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
? "%?"
?
0?????????)
? ?
0__inference_concatenate_58_layer_call_fn_2043535????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
? "??????????)?
E__inference_dense_59_layer_call_and_return_conditional_losses_2043546\z{/?,
%?"
 ?
inputs?????????)
? "%?"
?
0?????????

? }
*__inference_dense_59_layer_call_fn_2043555Oz{/?,
%?"
 ?
inputs?????????)
? "??????????
?
E__inference_dense_60_layer_call_and_return_conditional_losses_2043565^??/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? 
*__inference_dense_60_layer_call_fn_2043574Q??/?,
%?"
 ?
inputs?????????

? "???????????
J__inference_embedding_236_layer_call_and_return_conditional_losses_2043436_+/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????
? ?
/__inference_embedding_236_layer_call_fn_2043443R+/?,
%?"
 ?
inputs?????????	
? "???????????
J__inference_embedding_237_layer_call_and_return_conditional_losses_2043452_0/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????
? ?
/__inference_embedding_237_layer_call_fn_2043459R0/?,
%?"
 ?
inputs?????????	
? "???????????
J__inference_embedding_238_layer_call_and_return_conditional_losses_2043468_5/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????
? ?
/__inference_embedding_238_layer_call_fn_2043475R5/?,
%?"
 ?
inputs?????????	
? "???????????
G__inference_flatten_45_layer_call_and_return_conditional_losses_2043481\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? 
,__inference_flatten_45_layer_call_fn_2043486O3?0
)?&
$?!
inputs?????????
? "???????????
G__inference_flatten_46_layer_call_and_return_conditional_losses_2043492\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? 
,__inference_flatten_46_layer_call_fn_2043497O3?0
)?&
$?!
inputs?????????
? "???????????
G__inference_flatten_47_layer_call_and_return_conditional_losses_2043503\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? 
,__inference_flatten_47_layer_call_fn_2043508O3?0
)?&
$?!
inputs?????????
? "???????????
E__inference_model_29_layer_call_and_return_conditional_losses_2042937?*(?%?"?50+????????????z{?????
???
???
?
month?????????	
)?&
flat_model_type?????????
&?#
storey_range?????????
(?%
floor_area_sqm?????????
-?*
dist_to_nearest_stn?????????
'?$
dist_to_dhoby?????????
+?(
degree_centrality?????????
0?-
eigenvector_centrality?????????
/?,
remaining_lease_years?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_29_layer_call_and_return_conditional_losses_2043026?*(?%?"?50+????????????z{?????
???
???
?
month?????????	
)?&
flat_model_type?????????
&?#
storey_range?????????
(?%
floor_area_sqm?????????
-?*
dist_to_nearest_stn?????????
'?$
dist_to_dhoby?????????
+?(
degree_centrality?????????
0?-
eigenvector_centrality?????????
/?,
remaining_lease_years?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_model_29_layer_call_and_return_conditional_losses_2043199?*(?%?"?50+????????????z{?????
???
???
"?
inputs/0?????????	
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_29_layer_call_and_return_conditional_losses_2043301?*(?%?"?50+????????????z{?????
???
???
"?
inputs/0?????????	
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
p

 
? "%?"
?
0?????????
? ?
*__inference_model_29_layer_call_fn_2042494?*(?%?"?50+????????????z{?????
???
???
?
month?????????	
)?&
flat_model_type?????????
&?#
storey_range?????????
(?%
floor_area_sqm?????????
-?*
dist_to_nearest_stn?????????
'?$
dist_to_dhoby?????????
+?(
degree_centrality?????????
0?-
eigenvector_centrality?????????
/?,
remaining_lease_years?????????
p 

 
? "???????????
*__inference_model_29_layer_call_fn_2042848?*(?%?"?50+????????????z{?????
???
???
?
month?????????	
)?&
flat_model_type?????????
&?#
storey_range?????????
(?%
floor_area_sqm?????????
-?*
dist_to_nearest_stn?????????
'?$
dist_to_dhoby?????????
+?(
degree_centrality?????????
0?-
eigenvector_centrality?????????
/?,
remaining_lease_years?????????
p

 
? "???????????
*__inference_model_29_layer_call_fn_2043364?*(?%?"?50+????????????z{?????
???
???
"?
inputs/0?????????	
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
p 

 
? "???????????
*__inference_model_29_layer_call_fn_2043427?*(?%?"?50+????????????z{?????
???
???
"?
inputs/0?????????	
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
p

 
? "??????????{
__inference_restore_fn_2043700Y#K?H
A?>
?
restored_tensors_0	
?
restored_tensors_1	
? "? {
__inference_restore_fn_2043727Y&K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? {
__inference_restore_fn_2043754Y)K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_2043692?#&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor	
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
__inference_save_fn_2043719?&&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
__inference_save_fn_2043746?)&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
%__inference_signature_wrapper_2043097?*(?%?"?50+????????????z{?????
? 
???
@
degree_centrality+?(
degree_centrality?????????
8
dist_to_dhoby'?$
dist_to_dhoby?????????
D
dist_to_nearest_stn-?*
dist_to_nearest_stn?????????
J
eigenvector_centrality0?-
eigenvector_centrality?????????
<
flat_model_type)?&
flat_model_type?????????
:
floor_area_sqm(?%
floor_area_sqm?????????
(
month?
month?????????	
H
remaining_lease_years/?,
remaining_lease_years?????????
6
storey_range&?#
storey_range?????????"3?0
.
dense_60"?
dense_60?????????