??
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
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
o

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name	2075657*
value_dtype0	
?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_nametable_2075580*
value_dtype0	
q
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	2075881*
value_dtype0	
?
MutableHashTable_1MutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_2075804*
value_dtype0	
?
embedding_239/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameembedding_239/embeddings
?
,embedding_239/embeddings/Read/ReadVariableOpReadVariableOpembedding_239/embeddings*
_output_shapes

:*
dtype0
?
embedding_241/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameembedding_241/embeddings
?
,embedding_241/embeddings/Read/ReadVariableOpReadVariableOpembedding_241/embeddings*
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
|
dense_105/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*!
shared_namedense_105/kernel
u
$dense_105/kernel/Read/ReadVariableOpReadVariableOpdense_105/kernel*
_output_shapes

:
*
dtype0
t
dense_105/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_105/bias
m
"dense_105/bias/Read/ReadVariableOpReadVariableOpdense_105/bias*
_output_shapes
:
*
dtype0
|
dense_106/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*!
shared_namedense_106/kernel
u
$dense_106/kernel/Read/ReadVariableOpReadVariableOpdense_106/kernel*
_output_shapes

:
*
dtype0
t
dense_106/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_106/bias
m
"dense_106/bias/Read/ReadVariableOpReadVariableOpdense_106/bias*
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
Adam/embedding_239/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/embedding_239/embeddings/m
?
3Adam/embedding_239/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_239/embeddings/m*
_output_shapes

:*
dtype0
?
Adam/embedding_241/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/embedding_241/embeddings/m
?
3Adam/embedding_241/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_241/embeddings/m*
_output_shapes

:*
dtype0
?
Adam/dense_105/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_105/kernel/m
?
+Adam/dense_105/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_105/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_105/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_105/bias/m
{
)Adam/dense_105/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_105/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_106/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_106/kernel/m
?
+Adam/dense_106/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_106/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_106/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_106/bias/m
{
)Adam/dense_106/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_106/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_239/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/embedding_239/embeddings/v
?
3Adam/embedding_239/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_239/embeddings/v*
_output_shapes

:*
dtype0
?
Adam/embedding_241/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/embedding_241/embeddings/v
?
3Adam/embedding_241/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_241/embeddings/v*
_output_shapes

:*
dtype0
?
Adam/dense_105/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_105/kernel/v
?
+Adam/dense_105/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_105/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_105/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_105/bias/v
{
)Adam/dense_105/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_105/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_106/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_106/kernel/v
?
+Adam/dense_106/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_106/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_106/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_106/bias/v
{
)Adam/dense_106/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_106/bias/v*
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
\
Const_2Const*
_output_shapes

:*
dtype0*
valueB*xr?B
\
Const_3Const*
_output_shapes

:*
dtype0*
valueB**rD
\
Const_4Const*
_output_shapes

:*
dtype0*
valueB*}P?
\
Const_5Const*
_output_shapes

:*
dtype0*
valueB*d^>
\
Const_6Const*
_output_shapes

:*
dtype0*
valueB* ?3A
\
Const_7Const*
_output_shapes

:*
dtype0*
valueB*?z?A
\
Const_8Const*
_output_shapes

:*
dtype0*
valueB*`?<
\
Const_9Const*
_output_shapes

:*
dtype0*
valueB*s??7
]
Const_10Const*
_output_shapes

:*
dtype0*
valueB*>r?;
]
Const_11Const*
_output_shapes

:*
dtype0*
valueB*???9
]
Const_12Const*
_output_shapes

:*
dtype0*
valueB*z?B
]
Const_13Const*
_output_shapes

:*
dtype0*
valueB*R?'C
J
Const_14Const*
_output_shapes
: *
dtype0	*
value	B	 R 
J
Const_15Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Const_16Const*
_output_shapes
:*
dtype0	*u
valuelBj	"`       
                            	                                                 
?
Const_17Const*
_output_shapes
:*
dtype0	*u
valuelBj	"`                                                        	       
                     
?
Const_18Const*
_output_shapes
:*
dtype0*?
value?B?B04 TO 06B07 TO 09B10 TO 12B01 TO 03B13 TO 15B16 TO 18B19 TO 21B22 TO 24B25 TO 27B28 TO 30B31 TO 33B37 TO 39B34 TO 36B40 TO 42B46 TO 48B43 TO 45B49 TO 51
?
Const_19Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                        	       
                                                        
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_16Const_17*
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
__inference_<lambda>_2857867
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
__inference_<lambda>_2857872
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_1Const_18Const_19*
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
__inference_<lambda>_2857880
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
__inference_<lambda>_2857885
h
NoOpNoOp^PartitionedCall^PartitionedCall_1^StatefulPartitionedCall^StatefulPartitionedCall_1
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
?Q
Const_20Const"/device:CPU:0*
_output_shapes
: *
dtype0*?Q
value?QB?Q B?Q
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
layer_with_weights-5
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer_with_weights-8
layer-18
layer_with_weights-9
layer-19
layer-20
layer_with_weights-10
layer-21
layer-22
layer_with_weights-11
layer-23
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
 
3
lookup_table
 token_counts
!	keras_api
3
"lookup_table
#token_counts
$	keras_api
b
%
embeddings
&regularization_losses
'	variables
(trainable_variables
)	keras_api
b
*
embeddings
+regularization_losses
,	variables
-trainable_variables
.	keras_api
 
 
 
 
 
 
R
/regularization_losses
0	variables
1trainable_variables
2	keras_api
R
3regularization_losses
4	variables
5trainable_variables
6	keras_api
?
7
_keep_axis
8_reduce_axis
9_reduce_axis_mask
:_broadcast_shape
;mean
;
adapt_mean
<variance
<adapt_variance
	=count
>	keras_api
?
?
_keep_axis
@_reduce_axis
A_reduce_axis_mask
B_broadcast_shape
Cmean
C
adapt_mean
Dvariance
Dadapt_variance
	Ecount
F	keras_api
?
G
_keep_axis
H_reduce_axis
I_reduce_axis_mask
J_broadcast_shape
Kmean
K
adapt_mean
Lvariance
Ladapt_variance
	Mcount
N	keras_api
?
O
_keep_axis
P_reduce_axis
Q_reduce_axis_mask
R_broadcast_shape
Smean
S
adapt_mean
Tvariance
Tadapt_variance
	Ucount
V	keras_api
?
W
_keep_axis
X_reduce_axis
Y_reduce_axis_mask
Z_broadcast_shape
[mean
[
adapt_mean
\variance
\adapt_variance
	]count
^	keras_api
?
_
_keep_axis
`_reduce_axis
a_reduce_axis_mask
b_broadcast_shape
cmean
c
adapt_mean
dvariance
dadapt_variance
	ecount
f	keras_api
R
gregularization_losses
h	variables
itrainable_variables
j	keras_api
h

kkernel
lbias
mregularization_losses
n	variables
otrainable_variables
p	keras_api
 
h

qkernel
rbias
sregularization_losses
t	variables
utrainable_variables
v	keras_api
?
witer

xbeta_1

ybeta_2
	zdecay
{learning_rate%m?*m?km?lm?qm?rm?%v?*v?kv?lv?qv?rv?
 
?
%2
*3
;4
<5
=6
C7
D8
E9
K10
L11
M12
S13
T14
U15
[16
\17
]18
c19
d20
e21
k22
l23
q24
r25
*
%0
*1
k2
l3
q4
r5
?
|layer_metrics

}layers
regularization_losses
~non_trainable_variables
metrics
	variables
 ?layer_regularization_losses
trainable_variables
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
hf
VARIABLE_VALUEembedding_239/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

%0

%0
?
?layer_metrics
?layers
&regularization_losses
?non_trainable_variables
?metrics
'	variables
 ?layer_regularization_losses
(trainable_variables
hf
VARIABLE_VALUEembedding_241/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

*0

*0
?
?layer_metrics
?layers
+regularization_losses
?non_trainable_variables
?metrics
,	variables
 ?layer_regularization_losses
-trainable_variables
 
 
 
?
?layer_metrics
?layers
/regularization_losses
?non_trainable_variables
?metrics
0	variables
 ?layer_regularization_losses
1trainable_variables
 
 
 
?
?layer_metrics
?layers
3regularization_losses
?non_trainable_variables
?metrics
4	variables
 ?layer_regularization_losses
5trainable_variables
 
 
 
 
NL
VARIABLE_VALUEmean4layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvariance8layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcount5layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_14layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_18layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_15layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_24layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_28layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_25layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_34layer_with_weights-7/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_38layer_with_weights-7/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_35layer_with_weights-7/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_44layer_with_weights-8/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_48layer_with_weights-8/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_45layer_with_weights-8/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_54layer_with_weights-9/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_58layer_with_weights-9/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_55layer_with_weights-9/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
?
?layer_metrics
?layers
gregularization_losses
?non_trainable_variables
?metrics
h	variables
 ?layer_regularization_losses
itrainable_variables
][
VARIABLE_VALUEdense_105/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_105/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

k0
l1

k0
l1
?
?layer_metrics
?layers
mregularization_losses
?non_trainable_variables
?metrics
n	variables
 ?layer_regularization_losses
otrainable_variables
][
VARIABLE_VALUEdense_106/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_106/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

q0
r1
?
?layer_metrics
?layers
sregularization_losses
?non_trainable_variables
?metrics
t	variables
 ?layer_regularization_losses
utrainable_variables
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
?
;2
<3
=4
C5
D6
E7
K8
L9
M10
S11
T12
U13
[14
\15
]16
c17
d18
e19
 
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
VARIABLE_VALUEAdam/embedding_239/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_241/embeddings/mVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_105/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_105/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_106/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_106/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_239/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_241/embeddings/vVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_105/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_105/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_106/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_106/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCall_2StatefulPartitionedCall!serving_default_degree_centralityserving_default_dist_to_dhoby#serving_default_dist_to_nearest_stn&serving_default_eigenvector_centralityserving_default_flat_model_typeserving_default_floor_area_sqmserving_default_month%serving_default_remaining_lease_yearsserving_default_storey_rangehash_table_1Const
hash_tableConst_1embedding_241/embeddingsembedding_239/embeddingsConst_2Const_3Const_4Const_5Const_6Const_7Const_8Const_9Const_10Const_11Const_12Const_13dense_105/kerneldense_105/biasdense_106/kerneldense_106/bias**
Tin#
!2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_2857325
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1AMutableHashTable_1_lookup_table_export_values/LookupTableExportV2CMutableHashTable_1_lookup_table_export_values/LookupTableExportV2:1,embedding_239/embeddings/Read/ReadVariableOp,embedding_241/embeddings/Read/ReadVariableOpmean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOpmean_1/Read/ReadVariableOpvariance_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpmean_2/Read/ReadVariableOpvariance_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpmean_3/Read/ReadVariableOpvariance_3/Read/ReadVariableOpcount_3/Read/ReadVariableOpmean_4/Read/ReadVariableOpvariance_4/Read/ReadVariableOpcount_4/Read/ReadVariableOpmean_5/Read/ReadVariableOpvariance_5/Read/ReadVariableOpcount_5/Read/ReadVariableOp$dense_105/kernel/Read/ReadVariableOp"dense_105/bias/Read/ReadVariableOp$dense_106/kernel/Read/ReadVariableOp"dense_106/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount_6/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_8/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_9/Read/ReadVariableOp3Adam/embedding_239/embeddings/m/Read/ReadVariableOp3Adam/embedding_241/embeddings/m/Read/ReadVariableOp+Adam/dense_105/kernel/m/Read/ReadVariableOp)Adam/dense_105/bias/m/Read/ReadVariableOp+Adam/dense_106/kernel/m/Read/ReadVariableOp)Adam/dense_106/bias/m/Read/ReadVariableOp3Adam/embedding_239/embeddings/v/Read/ReadVariableOp3Adam/embedding_241/embeddings/v/Read/ReadVariableOp+Adam/dense_105/kernel/v/Read/ReadVariableOp)Adam/dense_105/bias/v/Read/ReadVariableOp+Adam/dense_106/kernel/v/Read/ReadVariableOp)Adam/dense_106/bias/v/Read/ReadVariableOpConst_20*B
Tin;
927										*
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
 __inference__traced_save_2858099
?
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filenameMutableHashTableMutableHashTable_1embedding_239/embeddingsembedding_241/embeddingsmeanvariancecountmean_1
variance_1count_1mean_2
variance_2count_2mean_3
variance_3count_3mean_4
variance_4count_4mean_5
variance_5count_5dense_105/kerneldense_105/biasdense_106/kerneldense_106/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount_6total_1count_7total_2count_8total_3count_9Adam/embedding_239/embeddings/mAdam/embedding_241/embeddings/mAdam/dense_105/kernel/mAdam/dense_105/bias/mAdam/dense_106/kernel/mAdam/dense_106/bias/mAdam/embedding_239/embeddings/vAdam/embedding_241/embeddings/vAdam/dense_105/kernel/vAdam/dense_105/bias/vAdam/dense_106/kernel/vAdam/dense_106/bias/v*?
Tin8
624*
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
#__inference__traced_restore_2858262ܙ
?
c
G__inference_flatten_50_layer_call_and_return_conditional_losses_2857670

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
?
?
*__inference_model_52_layer_call_fn_2857564
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
	unknown_2	
	unknown_3:
	unknown_4:
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17:


unknown_18:


unknown_19:


unknown_20:
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
unknown_20**
Tin#
!2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_52_layer_call_and_return_conditional_losses_28567382
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
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 22
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
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

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

:
?
?
*__inference_model_52_layer_call_fn_2857621
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
	unknown_2	
	unknown_3:
	unknown_4:
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17:


unknown_18:


unknown_19:


unknown_20:
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
unknown_20**
Tin#
!2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_52_layer_call_and_return_conditional_losses_28569942
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
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 22
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
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

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

:
?
?
%__inference_signature_wrapper_2857325
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
	unknown_2	
	unknown_3:
	unknown_4:
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17:


unknown_18:


unknown_19:


unknown_20:
identity??StatefulPartitionedCall?
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
unknown_20**
Tin#
!2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_28565742
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
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 22
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
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

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

:
?
?
__inference_<lambda>_2857867:
6key_value_init2075656_lookuptableimportv2_table_handle2
.key_value_init2075656_lookuptableimportv2_keys	4
0key_value_init2075656_lookuptableimportv2_values	
identity??)key_value_init2075656/LookupTableImportV2?
)key_value_init2075656/LookupTableImportV2LookupTableImportV26key_value_init2075656_lookuptableimportv2_table_handle.key_value_init2075656_lookuptableimportv2_keys0key_value_init2075656_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 2+
)key_value_init2075656/LookupTableImportV2S
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
NoOpNoOp*^key_value_init2075656/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init2075656/LookupTableImportV2)key_value_init2075656/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
.
__inference__destroyer_2857772
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
J__inference_embedding_241_layer_call_and_return_conditional_losses_2856614

inputs	*
embedding_lookup_2856608:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2856608inputs",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/2856608*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@embedding_lookup/2856608*+
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
?
?
*__inference_model_52_layer_call_fn_2856785	
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
	unknown_2	
	unknown_3:
	unknown_4:
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17:


unknown_18:


unknown_19:


unknown_20:
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
unknown_20**
Tin#
!2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_52_layer_call_and_return_conditional_losses_28567382
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
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 22
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
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

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

:
?n
?
E__inference_model_52_layer_call_and_return_conditional_losses_2857507
inputs_0	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8E
Astring_lookup_87_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_87_hash_table_lookup_lookuptablefindv2_default_value	G
Cinteger_lookup_131_hash_table_lookup_lookuptablefindv2_table_handleH
Dinteger_lookup_131_hash_table_lookup_lookuptablefindv2_default_value	8
&embedding_241_embedding_lookup_2857435:8
&embedding_239_embedding_lookup_2857440:
normalization_354_sub_y
normalization_354_sqrt_x
normalization_355_sub_y
normalization_355_sqrt_x
normalization_356_sub_y
normalization_356_sqrt_x
normalization_357_sub_y
normalization_357_sqrt_x
normalization_358_sub_y
normalization_358_sqrt_x
normalization_359_sub_y
normalization_359_sqrt_x:
(dense_105_matmul_readvariableop_resource:
7
)dense_105_biasadd_readvariableop_resource:
:
(dense_106_matmul_readvariableop_resource:
7
)dense_106_biasadd_readvariableop_resource:
identity?? dense_105/BiasAdd/ReadVariableOp?dense_105/MatMul/ReadVariableOp? dense_106/BiasAdd/ReadVariableOp?dense_106/MatMul/ReadVariableOp?embedding_239/embedding_lookup?embedding_241/embedding_lookup?6integer_lookup_131/hash_table_Lookup/LookupTableFindV2?4string_lookup_87/hash_table_Lookup/LookupTableFindV2?
4string_lookup_87/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_87_hash_table_lookup_lookuptablefindv2_table_handleinputs_2Bstring_lookup_87_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_87/hash_table_Lookup/LookupTableFindV2?
string_lookup_87/IdentityIdentity=string_lookup_87/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_87/Identity?
6integer_lookup_131/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Cinteger_lookup_131_hash_table_lookup_lookuptablefindv2_table_handleinputs_0Dinteger_lookup_131_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????28
6integer_lookup_131/hash_table_Lookup/LookupTableFindV2?
integer_lookup_131/IdentityIdentity?integer_lookup_131/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup_131/Identity?
embedding_241/embedding_lookupResourceGather&embedding_241_embedding_lookup_2857435"string_lookup_87/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*9
_class/
-+loc:@embedding_241/embedding_lookup/2857435*+
_output_shapes
:?????????*
dtype02 
embedding_241/embedding_lookup?
'embedding_241/embedding_lookup/IdentityIdentity'embedding_241/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*9
_class/
-+loc:@embedding_241/embedding_lookup/2857435*+
_output_shapes
:?????????2)
'embedding_241/embedding_lookup/Identity?
)embedding_241/embedding_lookup/Identity_1Identity0embedding_241/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2+
)embedding_241/embedding_lookup/Identity_1?
embedding_239/embedding_lookupResourceGather&embedding_239_embedding_lookup_2857440$integer_lookup_131/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*9
_class/
-+loc:@embedding_239/embedding_lookup/2857440*+
_output_shapes
:?????????*
dtype02 
embedding_239/embedding_lookup?
'embedding_239/embedding_lookup/IdentityIdentity'embedding_239/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*9
_class/
-+loc:@embedding_239/embedding_lookup/2857440*+
_output_shapes
:?????????2)
'embedding_239/embedding_lookup/Identity?
)embedding_239/embedding_lookup/Identity_1Identity0embedding_239/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2+
)embedding_239/embedding_lookup/Identity_1u
flatten_48/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_48/Const?
flatten_48/ReshapeReshape2embedding_239/embedding_lookup/Identity_1:output:0flatten_48/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_48/Reshapeu
flatten_50/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_50/Const?
flatten_50/ReshapeReshape2embedding_241/embedding_lookup/Identity_1:output:0flatten_50/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_50/Reshape?
normalization_354/subSubinputs_3normalization_354_sub_y*
T0*'
_output_shapes
:?????????2
normalization_354/sub{
normalization_354/SqrtSqrtnormalization_354_sqrt_x*
T0*
_output_shapes

:2
normalization_354/Sqrt
normalization_354/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_354/Maximum/y?
normalization_354/MaximumMaximumnormalization_354/Sqrt:y:0$normalization_354/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_354/Maximum?
normalization_354/truedivRealDivnormalization_354/sub:z:0normalization_354/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_354/truediv?
normalization_355/subSubinputs_4normalization_355_sub_y*
T0*'
_output_shapes
:?????????2
normalization_355/sub{
normalization_355/SqrtSqrtnormalization_355_sqrt_x*
T0*
_output_shapes

:2
normalization_355/Sqrt
normalization_355/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_355/Maximum/y?
normalization_355/MaximumMaximumnormalization_355/Sqrt:y:0$normalization_355/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_355/Maximum?
normalization_355/truedivRealDivnormalization_355/sub:z:0normalization_355/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_355/truediv?
normalization_356/subSubinputs_5normalization_356_sub_y*
T0*'
_output_shapes
:?????????2
normalization_356/sub{
normalization_356/SqrtSqrtnormalization_356_sqrt_x*
T0*
_output_shapes

:2
normalization_356/Sqrt
normalization_356/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_356/Maximum/y?
normalization_356/MaximumMaximumnormalization_356/Sqrt:y:0$normalization_356/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_356/Maximum?
normalization_356/truedivRealDivnormalization_356/sub:z:0normalization_356/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_356/truediv?
normalization_357/subSubinputs_6normalization_357_sub_y*
T0*'
_output_shapes
:?????????2
normalization_357/sub{
normalization_357/SqrtSqrtnormalization_357_sqrt_x*
T0*
_output_shapes

:2
normalization_357/Sqrt
normalization_357/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_357/Maximum/y?
normalization_357/MaximumMaximumnormalization_357/Sqrt:y:0$normalization_357/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_357/Maximum?
normalization_357/truedivRealDivnormalization_357/sub:z:0normalization_357/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_357/truediv?
normalization_358/subSubinputs_7normalization_358_sub_y*
T0*'
_output_shapes
:?????????2
normalization_358/sub{
normalization_358/SqrtSqrtnormalization_358_sqrt_x*
T0*
_output_shapes

:2
normalization_358/Sqrt
normalization_358/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_358/Maximum/y?
normalization_358/MaximumMaximumnormalization_358/Sqrt:y:0$normalization_358/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_358/Maximum?
normalization_358/truedivRealDivnormalization_358/sub:z:0normalization_358/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_358/truediv?
normalization_359/subSubinputs_8normalization_359_sub_y*
T0*'
_output_shapes
:?????????2
normalization_359/sub{
normalization_359/SqrtSqrtnormalization_359_sqrt_x*
T0*
_output_shapes

:2
normalization_359/Sqrt
normalization_359/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_359/Maximum/y?
normalization_359/MaximumMaximumnormalization_359/Sqrt:y:0$normalization_359/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_359/Maximum?
normalization_359/truedivRealDivnormalization_359/sub:z:0normalization_359/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_359/truedivz
concatenate_81/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_81/concat/axis?
concatenate_81/concatConcatV2flatten_48/Reshape:output:0flatten_50/Reshape:output:0normalization_354/truediv:z:0normalization_355/truediv:z:0normalization_356/truediv:z:0normalization_357/truediv:z:0normalization_358/truediv:z:0normalization_359/truediv:z:0#concatenate_81/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_81/concat?
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02!
dense_105/MatMul/ReadVariableOp?
dense_105/MatMulMatMulconcatenate_81/concat:output:0'dense_105/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_105/MatMul?
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_105/BiasAdd/ReadVariableOp?
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_105/BiasAddv
dense_105/ReluReludense_105/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_105/Relu?
dense_106/MatMul/ReadVariableOpReadVariableOp(dense_106_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02!
dense_106/MatMul/ReadVariableOp?
dense_106/MatMulMatMuldense_105/Relu:activations:0'dense_106/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_106/MatMul?
 dense_106/BiasAdd/ReadVariableOpReadVariableOp)dense_106_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_106/BiasAdd/ReadVariableOp?
dense_106/BiasAddBiasAdddense_106/MatMul:product:0(dense_106/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_106/BiasAddu
IdentityIdentitydense_106/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp!^dense_106/BiasAdd/ReadVariableOp ^dense_106/MatMul/ReadVariableOp^embedding_239/embedding_lookup^embedding_241/embedding_lookup7^integer_lookup_131/hash_table_Lookup/LookupTableFindV25^string_lookup_87/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2D
 dense_106/BiasAdd/ReadVariableOp dense_106/BiasAdd/ReadVariableOp2B
dense_106/MatMul/ReadVariableOpdense_106/MatMul/ReadVariableOp2@
embedding_239/embedding_lookupembedding_239/embedding_lookup2@
embedding_241/embedding_lookupembedding_241/embedding_lookup2p
6integer_lookup_131/hash_table_Lookup/LookupTableFindV26integer_lookup_131/hash_table_Lookup/LookupTableFindV22l
4string_lookup_87/hash_table_Lookup/LookupTableFindV24string_lookup_87/hash_table_Lookup/LookupTableFindV2:Q M
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
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

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

:
?

?
J__inference_embedding_239_layer_call_and_return_conditional_losses_2856627

inputs	*
embedding_lookup_2856621:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2856621inputs",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/2856621*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@embedding_lookup/2856621*+
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
?,
?
__inference_adapt_step_2083294
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
K__inference_concatenate_81_layer_call_and_return_conditional_losses_2857688
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
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
inputs/7
?
H
,__inference_flatten_50_layer_call_fn_2857675

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
G__inference_flatten_50_layer_call_and_return_conditional_losses_28566452
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
?
.
__inference__destroyer_2857757
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
?
?
__inference_save_fn_2857824
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
?
0
 __inference__initializer_2857767
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
__inference_restore_fn_2857832
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
?c
?

E__inference_model_52_layer_call_and_return_conditional_losses_2856738

inputs	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8E
Astring_lookup_87_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_87_hash_table_lookup_lookuptablefindv2_default_value	G
Cinteger_lookup_131_hash_table_lookup_lookuptablefindv2_table_handleH
Dinteger_lookup_131_hash_table_lookup_lookuptablefindv2_default_value	'
embedding_241_2856615:'
embedding_239_2856628:
normalization_354_sub_y
normalization_354_sqrt_x
normalization_355_sub_y
normalization_355_sqrt_x
normalization_356_sub_y
normalization_356_sqrt_x
normalization_357_sub_y
normalization_357_sqrt_x
normalization_358_sub_y
normalization_358_sqrt_x
normalization_359_sub_y
normalization_359_sqrt_x#
dense_105_2856716:

dense_105_2856718:
#
dense_106_2856732:

dense_106_2856734:
identity??!dense_105/StatefulPartitionedCall?!dense_106/StatefulPartitionedCall?%embedding_239/StatefulPartitionedCall?%embedding_241/StatefulPartitionedCall?6integer_lookup_131/hash_table_Lookup/LookupTableFindV2?4string_lookup_87/hash_table_Lookup/LookupTableFindV2?
4string_lookup_87/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_87_hash_table_lookup_lookuptablefindv2_table_handleinputs_2Bstring_lookup_87_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_87/hash_table_Lookup/LookupTableFindV2?
string_lookup_87/IdentityIdentity=string_lookup_87/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_87/Identity?
6integer_lookup_131/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Cinteger_lookup_131_hash_table_lookup_lookuptablefindv2_table_handleinputsDinteger_lookup_131_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????28
6integer_lookup_131/hash_table_Lookup/LookupTableFindV2?
integer_lookup_131/IdentityIdentity?integer_lookup_131/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup_131/Identity?
%embedding_241/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_87/Identity:output:0embedding_241_2856615*
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
J__inference_embedding_241_layer_call_and_return_conditional_losses_28566142'
%embedding_241/StatefulPartitionedCall?
%embedding_239/StatefulPartitionedCallStatefulPartitionedCall$integer_lookup_131/Identity:output:0embedding_239_2856628*
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
J__inference_embedding_239_layer_call_and_return_conditional_losses_28566272'
%embedding_239/StatefulPartitionedCall?
flatten_48/PartitionedCallPartitionedCall.embedding_239/StatefulPartitionedCall:output:0*
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
G__inference_flatten_48_layer_call_and_return_conditional_losses_28566372
flatten_48/PartitionedCall?
flatten_50/PartitionedCallPartitionedCall.embedding_241/StatefulPartitionedCall:output:0*
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
G__inference_flatten_50_layer_call_and_return_conditional_losses_28566452
flatten_50/PartitionedCall?
normalization_354/subSubinputs_3normalization_354_sub_y*
T0*'
_output_shapes
:?????????2
normalization_354/sub{
normalization_354/SqrtSqrtnormalization_354_sqrt_x*
T0*
_output_shapes

:2
normalization_354/Sqrt
normalization_354/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_354/Maximum/y?
normalization_354/MaximumMaximumnormalization_354/Sqrt:y:0$normalization_354/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_354/Maximum?
normalization_354/truedivRealDivnormalization_354/sub:z:0normalization_354/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_354/truediv?
normalization_355/subSubinputs_4normalization_355_sub_y*
T0*'
_output_shapes
:?????????2
normalization_355/sub{
normalization_355/SqrtSqrtnormalization_355_sqrt_x*
T0*
_output_shapes

:2
normalization_355/Sqrt
normalization_355/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_355/Maximum/y?
normalization_355/MaximumMaximumnormalization_355/Sqrt:y:0$normalization_355/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_355/Maximum?
normalization_355/truedivRealDivnormalization_355/sub:z:0normalization_355/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_355/truediv?
normalization_356/subSubinputs_5normalization_356_sub_y*
T0*'
_output_shapes
:?????????2
normalization_356/sub{
normalization_356/SqrtSqrtnormalization_356_sqrt_x*
T0*
_output_shapes

:2
normalization_356/Sqrt
normalization_356/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_356/Maximum/y?
normalization_356/MaximumMaximumnormalization_356/Sqrt:y:0$normalization_356/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_356/Maximum?
normalization_356/truedivRealDivnormalization_356/sub:z:0normalization_356/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_356/truediv?
normalization_357/subSubinputs_6normalization_357_sub_y*
T0*'
_output_shapes
:?????????2
normalization_357/sub{
normalization_357/SqrtSqrtnormalization_357_sqrt_x*
T0*
_output_shapes

:2
normalization_357/Sqrt
normalization_357/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_357/Maximum/y?
normalization_357/MaximumMaximumnormalization_357/Sqrt:y:0$normalization_357/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_357/Maximum?
normalization_357/truedivRealDivnormalization_357/sub:z:0normalization_357/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_357/truediv?
normalization_358/subSubinputs_7normalization_358_sub_y*
T0*'
_output_shapes
:?????????2
normalization_358/sub{
normalization_358/SqrtSqrtnormalization_358_sqrt_x*
T0*
_output_shapes

:2
normalization_358/Sqrt
normalization_358/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_358/Maximum/y?
normalization_358/MaximumMaximumnormalization_358/Sqrt:y:0$normalization_358/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_358/Maximum?
normalization_358/truedivRealDivnormalization_358/sub:z:0normalization_358/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_358/truediv?
normalization_359/subSubinputs_8normalization_359_sub_y*
T0*'
_output_shapes
:?????????2
normalization_359/sub{
normalization_359/SqrtSqrtnormalization_359_sqrt_x*
T0*
_output_shapes

:2
normalization_359/Sqrt
normalization_359/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_359/Maximum/y?
normalization_359/MaximumMaximumnormalization_359/Sqrt:y:0$normalization_359/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_359/Maximum?
normalization_359/truedivRealDivnormalization_359/sub:z:0normalization_359/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_359/truediv?
concatenate_81/PartitionedCallPartitionedCall#flatten_48/PartitionedCall:output:0#flatten_50/PartitionedCall:output:0normalization_354/truediv:z:0normalization_355/truediv:z:0normalization_356/truediv:z:0normalization_357/truediv:z:0normalization_358/truediv:z:0normalization_359/truediv:z:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_concatenate_81_layer_call_and_return_conditional_losses_28567022 
concatenate_81/PartitionedCall?
!dense_105/StatefulPartitionedCallStatefulPartitionedCall'concatenate_81/PartitionedCall:output:0dense_105_2856716dense_105_2856718*
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
GPU2*0J 8? *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_28567152#
!dense_105/StatefulPartitionedCall?
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0dense_106_2856732dense_106_2856734*
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
GPU2*0J 8? *O
fJRH
F__inference_dense_106_layer_call_and_return_conditional_losses_28567312#
!dense_106/StatefulPartitionedCall?
IdentityIdentity*dense_106/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall&^embedding_239/StatefulPartitionedCall&^embedding_241/StatefulPartitionedCall7^integer_lookup_131/hash_table_Lookup/LookupTableFindV25^string_lookup_87/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2N
%embedding_239/StatefulPartitionedCall%embedding_239/StatefulPartitionedCall2N
%embedding_241/StatefulPartitionedCall%embedding_241/StatefulPartitionedCall2p
6integer_lookup_131/hash_table_Lookup/LookupTableFindV26integer_lookup_131/hash_table_Lookup/LookupTableFindV22l
4string_lookup_87/hash_table_Lookup/LookupTableFindV24string_lookup_87/hash_table_Lookup/LookupTableFindV2:O K
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
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

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

:
?
?
K__inference_concatenate_81_layer_call_and_return_conditional_losses_2856702

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
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
 
_user_specified_nameinputs
?
?
__inference_<lambda>_2857880:
6key_value_init2075880_lookuptableimportv2_table_handle2
.key_value_init2075880_lookuptableimportv2_keys4
0key_value_init2075880_lookuptableimportv2_values	
identity??)key_value_init2075880/LookupTableImportV2?
)key_value_init2075880/LookupTableImportV2LookupTableImportV26key_value_init2075880_lookuptableimportv2_table_handle.key_value_init2075880_lookuptableimportv2_keys0key_value_init2075880_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2+
)key_value_init2075880/LookupTableImportV2S
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
NoOpNoOp*^key_value_init2075880/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init2075880/LookupTableImportV2)key_value_init2075880/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
 __inference__initializer_2857785:
6key_value_init2075880_lookuptableimportv2_table_handle2
.key_value_init2075880_lookuptableimportv2_keys4
0key_value_init2075880_lookuptableimportv2_values	
identity??)key_value_init2075880/LookupTableImportV2?
)key_value_init2075880/LookupTableImportV2LookupTableImportV26key_value_init2075880_lookuptableimportv2_table_handle.key_value_init2075880_lookuptableimportv2_keys0key_value_init2075880_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2+
)key_value_init2075880/LookupTableImportV2P
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
NoOpNoOp*^key_value_init2075880/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init2075880/LookupTableImportV2)key_value_init2075880/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
+__inference_dense_106_layer_call_fn_2857739

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
GPU2*0J 8? *O
fJRH
F__inference_dense_106_layer_call_and_return_conditional_losses_28567312
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
?e
?
E__inference_model_52_layer_call_and_return_conditional_losses_2857260	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_yearsE
Astring_lookup_87_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_87_hash_table_lookup_lookuptablefindv2_default_value	G
Cinteger_lookup_131_hash_table_lookup_lookuptablefindv2_table_handleH
Dinteger_lookup_131_hash_table_lookup_lookuptablefindv2_default_value	'
embedding_241_2857198:'
embedding_239_2857201:
normalization_354_sub_y
normalization_354_sqrt_x
normalization_355_sub_y
normalization_355_sqrt_x
normalization_356_sub_y
normalization_356_sqrt_x
normalization_357_sub_y
normalization_357_sqrt_x
normalization_358_sub_y
normalization_358_sqrt_x
normalization_359_sub_y
normalization_359_sqrt_x#
dense_105_2857249:

dense_105_2857251:
#
dense_106_2857254:

dense_106_2857256:
identity??!dense_105/StatefulPartitionedCall?!dense_106/StatefulPartitionedCall?%embedding_239/StatefulPartitionedCall?%embedding_241/StatefulPartitionedCall?6integer_lookup_131/hash_table_Lookup/LookupTableFindV2?4string_lookup_87/hash_table_Lookup/LookupTableFindV2?
4string_lookup_87/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_87_hash_table_lookup_lookuptablefindv2_table_handlestorey_rangeBstring_lookup_87_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_87/hash_table_Lookup/LookupTableFindV2?
string_lookup_87/IdentityIdentity=string_lookup_87/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_87/Identity?
6integer_lookup_131/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Cinteger_lookup_131_hash_table_lookup_lookuptablefindv2_table_handlemonthDinteger_lookup_131_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????28
6integer_lookup_131/hash_table_Lookup/LookupTableFindV2?
integer_lookup_131/IdentityIdentity?integer_lookup_131/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup_131/Identity?
%embedding_241/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_87/Identity:output:0embedding_241_2857198*
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
J__inference_embedding_241_layer_call_and_return_conditional_losses_28566142'
%embedding_241/StatefulPartitionedCall?
%embedding_239/StatefulPartitionedCallStatefulPartitionedCall$integer_lookup_131/Identity:output:0embedding_239_2857201*
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
J__inference_embedding_239_layer_call_and_return_conditional_losses_28566272'
%embedding_239/StatefulPartitionedCall?
flatten_48/PartitionedCallPartitionedCall.embedding_239/StatefulPartitionedCall:output:0*
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
G__inference_flatten_48_layer_call_and_return_conditional_losses_28566372
flatten_48/PartitionedCall?
flatten_50/PartitionedCallPartitionedCall.embedding_241/StatefulPartitionedCall:output:0*
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
G__inference_flatten_50_layer_call_and_return_conditional_losses_28566452
flatten_50/PartitionedCall?
normalization_354/subSubfloor_area_sqmnormalization_354_sub_y*
T0*'
_output_shapes
:?????????2
normalization_354/sub{
normalization_354/SqrtSqrtnormalization_354_sqrt_x*
T0*
_output_shapes

:2
normalization_354/Sqrt
normalization_354/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_354/Maximum/y?
normalization_354/MaximumMaximumnormalization_354/Sqrt:y:0$normalization_354/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_354/Maximum?
normalization_354/truedivRealDivnormalization_354/sub:z:0normalization_354/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_354/truediv?
normalization_355/subSubdist_to_nearest_stnnormalization_355_sub_y*
T0*'
_output_shapes
:?????????2
normalization_355/sub{
normalization_355/SqrtSqrtnormalization_355_sqrt_x*
T0*
_output_shapes

:2
normalization_355/Sqrt
normalization_355/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_355/Maximum/y?
normalization_355/MaximumMaximumnormalization_355/Sqrt:y:0$normalization_355/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_355/Maximum?
normalization_355/truedivRealDivnormalization_355/sub:z:0normalization_355/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_355/truediv?
normalization_356/subSubdist_to_dhobynormalization_356_sub_y*
T0*'
_output_shapes
:?????????2
normalization_356/sub{
normalization_356/SqrtSqrtnormalization_356_sqrt_x*
T0*
_output_shapes

:2
normalization_356/Sqrt
normalization_356/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_356/Maximum/y?
normalization_356/MaximumMaximumnormalization_356/Sqrt:y:0$normalization_356/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_356/Maximum?
normalization_356/truedivRealDivnormalization_356/sub:z:0normalization_356/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_356/truediv?
normalization_357/subSubdegree_centralitynormalization_357_sub_y*
T0*'
_output_shapes
:?????????2
normalization_357/sub{
normalization_357/SqrtSqrtnormalization_357_sqrt_x*
T0*
_output_shapes

:2
normalization_357/Sqrt
normalization_357/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_357/Maximum/y?
normalization_357/MaximumMaximumnormalization_357/Sqrt:y:0$normalization_357/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_357/Maximum?
normalization_357/truedivRealDivnormalization_357/sub:z:0normalization_357/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_357/truediv?
normalization_358/subSubeigenvector_centralitynormalization_358_sub_y*
T0*'
_output_shapes
:?????????2
normalization_358/sub{
normalization_358/SqrtSqrtnormalization_358_sqrt_x*
T0*
_output_shapes

:2
normalization_358/Sqrt
normalization_358/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_358/Maximum/y?
normalization_358/MaximumMaximumnormalization_358/Sqrt:y:0$normalization_358/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_358/Maximum?
normalization_358/truedivRealDivnormalization_358/sub:z:0normalization_358/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_358/truediv?
normalization_359/subSubremaining_lease_yearsnormalization_359_sub_y*
T0*'
_output_shapes
:?????????2
normalization_359/sub{
normalization_359/SqrtSqrtnormalization_359_sqrt_x*
T0*
_output_shapes

:2
normalization_359/Sqrt
normalization_359/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_359/Maximum/y?
normalization_359/MaximumMaximumnormalization_359/Sqrt:y:0$normalization_359/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_359/Maximum?
normalization_359/truedivRealDivnormalization_359/sub:z:0normalization_359/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_359/truediv?
concatenate_81/PartitionedCallPartitionedCall#flatten_48/PartitionedCall:output:0#flatten_50/PartitionedCall:output:0normalization_354/truediv:z:0normalization_355/truediv:z:0normalization_356/truediv:z:0normalization_357/truediv:z:0normalization_358/truediv:z:0normalization_359/truediv:z:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_concatenate_81_layer_call_and_return_conditional_losses_28567022 
concatenate_81/PartitionedCall?
!dense_105/StatefulPartitionedCallStatefulPartitionedCall'concatenate_81/PartitionedCall:output:0dense_105_2857249dense_105_2857251*
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
GPU2*0J 8? *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_28567152#
!dense_105/StatefulPartitionedCall?
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0dense_106_2857254dense_106_2857256*
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
GPU2*0J 8? *O
fJRH
F__inference_dense_106_layer_call_and_return_conditional_losses_28567312#
!dense_106/StatefulPartitionedCall?
IdentityIdentity*dense_106/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall&^embedding_239/StatefulPartitionedCall&^embedding_241/StatefulPartitionedCall7^integer_lookup_131/hash_table_Lookup/LookupTableFindV25^string_lookup_87/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2N
%embedding_239/StatefulPartitionedCall%embedding_239/StatefulPartitionedCall2N
%embedding_241/StatefulPartitionedCall%embedding_241/StatefulPartitionedCall2p
6integer_lookup_131/hash_table_Lookup/LookupTableFindV26integer_lookup_131/hash_table_Lookup/LookupTableFindV22l
4string_lookup_87/hash_table_Lookup/LookupTableFindV24string_lookup_87/hash_table_Lookup/LookupTableFindV2:N J
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
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

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

:
?c
?

E__inference_model_52_layer_call_and_return_conditional_losses_2856994

inputs	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8E
Astring_lookup_87_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_87_hash_table_lookup_lookuptablefindv2_default_value	G
Cinteger_lookup_131_hash_table_lookup_lookuptablefindv2_table_handleH
Dinteger_lookup_131_hash_table_lookup_lookuptablefindv2_default_value	'
embedding_241_2856932:'
embedding_239_2856935:
normalization_354_sub_y
normalization_354_sqrt_x
normalization_355_sub_y
normalization_355_sqrt_x
normalization_356_sub_y
normalization_356_sqrt_x
normalization_357_sub_y
normalization_357_sqrt_x
normalization_358_sub_y
normalization_358_sqrt_x
normalization_359_sub_y
normalization_359_sqrt_x#
dense_105_2856983:

dense_105_2856985:
#
dense_106_2856988:

dense_106_2856990:
identity??!dense_105/StatefulPartitionedCall?!dense_106/StatefulPartitionedCall?%embedding_239/StatefulPartitionedCall?%embedding_241/StatefulPartitionedCall?6integer_lookup_131/hash_table_Lookup/LookupTableFindV2?4string_lookup_87/hash_table_Lookup/LookupTableFindV2?
4string_lookup_87/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_87_hash_table_lookup_lookuptablefindv2_table_handleinputs_2Bstring_lookup_87_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_87/hash_table_Lookup/LookupTableFindV2?
string_lookup_87/IdentityIdentity=string_lookup_87/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_87/Identity?
6integer_lookup_131/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Cinteger_lookup_131_hash_table_lookup_lookuptablefindv2_table_handleinputsDinteger_lookup_131_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????28
6integer_lookup_131/hash_table_Lookup/LookupTableFindV2?
integer_lookup_131/IdentityIdentity?integer_lookup_131/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup_131/Identity?
%embedding_241/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_87/Identity:output:0embedding_241_2856932*
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
J__inference_embedding_241_layer_call_and_return_conditional_losses_28566142'
%embedding_241/StatefulPartitionedCall?
%embedding_239/StatefulPartitionedCallStatefulPartitionedCall$integer_lookup_131/Identity:output:0embedding_239_2856935*
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
J__inference_embedding_239_layer_call_and_return_conditional_losses_28566272'
%embedding_239/StatefulPartitionedCall?
flatten_48/PartitionedCallPartitionedCall.embedding_239/StatefulPartitionedCall:output:0*
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
G__inference_flatten_48_layer_call_and_return_conditional_losses_28566372
flatten_48/PartitionedCall?
flatten_50/PartitionedCallPartitionedCall.embedding_241/StatefulPartitionedCall:output:0*
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
G__inference_flatten_50_layer_call_and_return_conditional_losses_28566452
flatten_50/PartitionedCall?
normalization_354/subSubinputs_3normalization_354_sub_y*
T0*'
_output_shapes
:?????????2
normalization_354/sub{
normalization_354/SqrtSqrtnormalization_354_sqrt_x*
T0*
_output_shapes

:2
normalization_354/Sqrt
normalization_354/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_354/Maximum/y?
normalization_354/MaximumMaximumnormalization_354/Sqrt:y:0$normalization_354/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_354/Maximum?
normalization_354/truedivRealDivnormalization_354/sub:z:0normalization_354/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_354/truediv?
normalization_355/subSubinputs_4normalization_355_sub_y*
T0*'
_output_shapes
:?????????2
normalization_355/sub{
normalization_355/SqrtSqrtnormalization_355_sqrt_x*
T0*
_output_shapes

:2
normalization_355/Sqrt
normalization_355/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_355/Maximum/y?
normalization_355/MaximumMaximumnormalization_355/Sqrt:y:0$normalization_355/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_355/Maximum?
normalization_355/truedivRealDivnormalization_355/sub:z:0normalization_355/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_355/truediv?
normalization_356/subSubinputs_5normalization_356_sub_y*
T0*'
_output_shapes
:?????????2
normalization_356/sub{
normalization_356/SqrtSqrtnormalization_356_sqrt_x*
T0*
_output_shapes

:2
normalization_356/Sqrt
normalization_356/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_356/Maximum/y?
normalization_356/MaximumMaximumnormalization_356/Sqrt:y:0$normalization_356/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_356/Maximum?
normalization_356/truedivRealDivnormalization_356/sub:z:0normalization_356/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_356/truediv?
normalization_357/subSubinputs_6normalization_357_sub_y*
T0*'
_output_shapes
:?????????2
normalization_357/sub{
normalization_357/SqrtSqrtnormalization_357_sqrt_x*
T0*
_output_shapes

:2
normalization_357/Sqrt
normalization_357/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_357/Maximum/y?
normalization_357/MaximumMaximumnormalization_357/Sqrt:y:0$normalization_357/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_357/Maximum?
normalization_357/truedivRealDivnormalization_357/sub:z:0normalization_357/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_357/truediv?
normalization_358/subSubinputs_7normalization_358_sub_y*
T0*'
_output_shapes
:?????????2
normalization_358/sub{
normalization_358/SqrtSqrtnormalization_358_sqrt_x*
T0*
_output_shapes

:2
normalization_358/Sqrt
normalization_358/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_358/Maximum/y?
normalization_358/MaximumMaximumnormalization_358/Sqrt:y:0$normalization_358/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_358/Maximum?
normalization_358/truedivRealDivnormalization_358/sub:z:0normalization_358/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_358/truediv?
normalization_359/subSubinputs_8normalization_359_sub_y*
T0*'
_output_shapes
:?????????2
normalization_359/sub{
normalization_359/SqrtSqrtnormalization_359_sqrt_x*
T0*
_output_shapes

:2
normalization_359/Sqrt
normalization_359/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_359/Maximum/y?
normalization_359/MaximumMaximumnormalization_359/Sqrt:y:0$normalization_359/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_359/Maximum?
normalization_359/truedivRealDivnormalization_359/sub:z:0normalization_359/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_359/truediv?
concatenate_81/PartitionedCallPartitionedCall#flatten_48/PartitionedCall:output:0#flatten_50/PartitionedCall:output:0normalization_354/truediv:z:0normalization_355/truediv:z:0normalization_356/truediv:z:0normalization_357/truediv:z:0normalization_358/truediv:z:0normalization_359/truediv:z:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_concatenate_81_layer_call_and_return_conditional_losses_28567022 
concatenate_81/PartitionedCall?
!dense_105/StatefulPartitionedCallStatefulPartitionedCall'concatenate_81/PartitionedCall:output:0dense_105_2856983dense_105_2856985*
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
GPU2*0J 8? *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_28567152#
!dense_105/StatefulPartitionedCall?
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0dense_106_2856988dense_106_2856990*
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
GPU2*0J 8? *O
fJRH
F__inference_dense_106_layer_call_and_return_conditional_losses_28567312#
!dense_106/StatefulPartitionedCall?
IdentityIdentity*dense_106/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall&^embedding_239/StatefulPartitionedCall&^embedding_241/StatefulPartitionedCall7^integer_lookup_131/hash_table_Lookup/LookupTableFindV25^string_lookup_87/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2N
%embedding_239/StatefulPartitionedCall%embedding_239/StatefulPartitionedCall2N
%embedding_241/StatefulPartitionedCall%embedding_241/StatefulPartitionedCall2p
6integer_lookup_131/hash_table_Lookup/LookupTableFindV26integer_lookup_131/hash_table_Lookup/LookupTableFindV22l
4string_lookup_87/hash_table_Lookup/LookupTableFindV24string_lookup_87/hash_table_Lookup/LookupTableFindV2:O K
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
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

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

:
?

?
J__inference_embedding_239_layer_call_and_return_conditional_losses_2857630

inputs	*
embedding_lookup_2857624:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2857624inputs",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/2857624*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@embedding_lookup/2857624*+
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
?
?
/__inference_embedding_241_layer_call_fn_2857653

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
J__inference_embedding_241_layer_call_and_return_conditional_losses_28566142
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
?
H
__inference__creator_2857795
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_2075804*
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
?
?
__inference_save_fn_2857851
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
,
__inference_<lambda>_2857885
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
?
?
__inference_adapt_step_2083099
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
?,
?
__inference_adapt_step_2083247
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
?
<
__inference__creator_2857777
identity??
hash_table}

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	2075881*
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
 __inference__initializer_2857752:
6key_value_init2075656_lookuptableimportv2_table_handle2
.key_value_init2075656_lookuptableimportv2_keys	4
0key_value_init2075656_lookuptableimportv2_values	
identity??)key_value_init2075656/LookupTableImportV2?
)key_value_init2075656/LookupTableImportV2LookupTableImportV26key_value_init2075656_lookuptableimportv2_table_handle.key_value_init2075656_lookuptableimportv2_keys0key_value_init2075656_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 2+
)key_value_init2075656/LookupTableImportV2P
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
NoOpNoOp*^key_value_init2075656/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init2075656/LookupTableImportV2)key_value_init2075656/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
c
G__inference_flatten_50_layer_call_and_return_conditional_losses_2856645

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
?,
?
__inference_adapt_step_2083341
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
?
c
G__inference_flatten_48_layer_call_and_return_conditional_losses_2856637

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
?
?
*__inference_model_52_layer_call_fn_2857098	
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
	unknown_2	
	unknown_3:
	unknown_4:
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17:


unknown_18:


unknown_19:


unknown_20:
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
unknown_20**
Tin#
!2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_52_layer_call_and_return_conditional_losses_28569942
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
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 22
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
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

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

:
?
.
__inference__destroyer_2857805
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
J__inference_embedding_241_layer_call_and_return_conditional_losses_2857646

inputs	*
embedding_lookup_2857640:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2857640inputs",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/2857640*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@embedding_lookup/2857640*+
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
?
c
G__inference_flatten_48_layer_call_and_return_conditional_losses_2857659

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
,
__inference_<lambda>_2857872
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
?
?
/__inference_embedding_239_layer_call_fn_2857637

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
J__inference_embedding_239_layer_call_and_return_conditional_losses_28566272
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
?

?
F__inference_dense_106_layer_call_and_return_conditional_losses_2857730

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
?
?
__inference_adapt_step_2488642
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
?,
?
__inference_adapt_step_2083200
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
F__inference_dense_106_layer_call_and_return_conditional_losses_2856731

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
?
<
__inference__creator_2857744
identity??
hash_table}

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name	2075657*
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
__inference_restore_fn_2857859
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
?,
?
__inference_adapt_step_2083435
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
?n
?
E__inference_model_52_layer_call_and_return_conditional_losses_2857416
inputs_0	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8E
Astring_lookup_87_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_87_hash_table_lookup_lookuptablefindv2_default_value	G
Cinteger_lookup_131_hash_table_lookup_lookuptablefindv2_table_handleH
Dinteger_lookup_131_hash_table_lookup_lookuptablefindv2_default_value	8
&embedding_241_embedding_lookup_2857344:8
&embedding_239_embedding_lookup_2857349:
normalization_354_sub_y
normalization_354_sqrt_x
normalization_355_sub_y
normalization_355_sqrt_x
normalization_356_sub_y
normalization_356_sqrt_x
normalization_357_sub_y
normalization_357_sqrt_x
normalization_358_sub_y
normalization_358_sqrt_x
normalization_359_sub_y
normalization_359_sqrt_x:
(dense_105_matmul_readvariableop_resource:
7
)dense_105_biasadd_readvariableop_resource:
:
(dense_106_matmul_readvariableop_resource:
7
)dense_106_biasadd_readvariableop_resource:
identity?? dense_105/BiasAdd/ReadVariableOp?dense_105/MatMul/ReadVariableOp? dense_106/BiasAdd/ReadVariableOp?dense_106/MatMul/ReadVariableOp?embedding_239/embedding_lookup?embedding_241/embedding_lookup?6integer_lookup_131/hash_table_Lookup/LookupTableFindV2?4string_lookup_87/hash_table_Lookup/LookupTableFindV2?
4string_lookup_87/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_87_hash_table_lookup_lookuptablefindv2_table_handleinputs_2Bstring_lookup_87_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_87/hash_table_Lookup/LookupTableFindV2?
string_lookup_87/IdentityIdentity=string_lookup_87/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_87/Identity?
6integer_lookup_131/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Cinteger_lookup_131_hash_table_lookup_lookuptablefindv2_table_handleinputs_0Dinteger_lookup_131_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????28
6integer_lookup_131/hash_table_Lookup/LookupTableFindV2?
integer_lookup_131/IdentityIdentity?integer_lookup_131/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup_131/Identity?
embedding_241/embedding_lookupResourceGather&embedding_241_embedding_lookup_2857344"string_lookup_87/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*9
_class/
-+loc:@embedding_241/embedding_lookup/2857344*+
_output_shapes
:?????????*
dtype02 
embedding_241/embedding_lookup?
'embedding_241/embedding_lookup/IdentityIdentity'embedding_241/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*9
_class/
-+loc:@embedding_241/embedding_lookup/2857344*+
_output_shapes
:?????????2)
'embedding_241/embedding_lookup/Identity?
)embedding_241/embedding_lookup/Identity_1Identity0embedding_241/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2+
)embedding_241/embedding_lookup/Identity_1?
embedding_239/embedding_lookupResourceGather&embedding_239_embedding_lookup_2857349$integer_lookup_131/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*9
_class/
-+loc:@embedding_239/embedding_lookup/2857349*+
_output_shapes
:?????????*
dtype02 
embedding_239/embedding_lookup?
'embedding_239/embedding_lookup/IdentityIdentity'embedding_239/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*9
_class/
-+loc:@embedding_239/embedding_lookup/2857349*+
_output_shapes
:?????????2)
'embedding_239/embedding_lookup/Identity?
)embedding_239/embedding_lookup/Identity_1Identity0embedding_239/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2+
)embedding_239/embedding_lookup/Identity_1u
flatten_48/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_48/Const?
flatten_48/ReshapeReshape2embedding_239/embedding_lookup/Identity_1:output:0flatten_48/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_48/Reshapeu
flatten_50/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_50/Const?
flatten_50/ReshapeReshape2embedding_241/embedding_lookup/Identity_1:output:0flatten_50/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_50/Reshape?
normalization_354/subSubinputs_3normalization_354_sub_y*
T0*'
_output_shapes
:?????????2
normalization_354/sub{
normalization_354/SqrtSqrtnormalization_354_sqrt_x*
T0*
_output_shapes

:2
normalization_354/Sqrt
normalization_354/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_354/Maximum/y?
normalization_354/MaximumMaximumnormalization_354/Sqrt:y:0$normalization_354/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_354/Maximum?
normalization_354/truedivRealDivnormalization_354/sub:z:0normalization_354/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_354/truediv?
normalization_355/subSubinputs_4normalization_355_sub_y*
T0*'
_output_shapes
:?????????2
normalization_355/sub{
normalization_355/SqrtSqrtnormalization_355_sqrt_x*
T0*
_output_shapes

:2
normalization_355/Sqrt
normalization_355/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_355/Maximum/y?
normalization_355/MaximumMaximumnormalization_355/Sqrt:y:0$normalization_355/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_355/Maximum?
normalization_355/truedivRealDivnormalization_355/sub:z:0normalization_355/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_355/truediv?
normalization_356/subSubinputs_5normalization_356_sub_y*
T0*'
_output_shapes
:?????????2
normalization_356/sub{
normalization_356/SqrtSqrtnormalization_356_sqrt_x*
T0*
_output_shapes

:2
normalization_356/Sqrt
normalization_356/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_356/Maximum/y?
normalization_356/MaximumMaximumnormalization_356/Sqrt:y:0$normalization_356/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_356/Maximum?
normalization_356/truedivRealDivnormalization_356/sub:z:0normalization_356/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_356/truediv?
normalization_357/subSubinputs_6normalization_357_sub_y*
T0*'
_output_shapes
:?????????2
normalization_357/sub{
normalization_357/SqrtSqrtnormalization_357_sqrt_x*
T0*
_output_shapes

:2
normalization_357/Sqrt
normalization_357/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_357/Maximum/y?
normalization_357/MaximumMaximumnormalization_357/Sqrt:y:0$normalization_357/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_357/Maximum?
normalization_357/truedivRealDivnormalization_357/sub:z:0normalization_357/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_357/truediv?
normalization_358/subSubinputs_7normalization_358_sub_y*
T0*'
_output_shapes
:?????????2
normalization_358/sub{
normalization_358/SqrtSqrtnormalization_358_sqrt_x*
T0*
_output_shapes

:2
normalization_358/Sqrt
normalization_358/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_358/Maximum/y?
normalization_358/MaximumMaximumnormalization_358/Sqrt:y:0$normalization_358/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_358/Maximum?
normalization_358/truedivRealDivnormalization_358/sub:z:0normalization_358/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_358/truediv?
normalization_359/subSubinputs_8normalization_359_sub_y*
T0*'
_output_shapes
:?????????2
normalization_359/sub{
normalization_359/SqrtSqrtnormalization_359_sqrt_x*
T0*
_output_shapes

:2
normalization_359/Sqrt
normalization_359/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_359/Maximum/y?
normalization_359/MaximumMaximumnormalization_359/Sqrt:y:0$normalization_359/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_359/Maximum?
normalization_359/truedivRealDivnormalization_359/sub:z:0normalization_359/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_359/truedivz
concatenate_81/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_81/concat/axis?
concatenate_81/concatConcatV2flatten_48/Reshape:output:0flatten_50/Reshape:output:0normalization_354/truediv:z:0normalization_355/truediv:z:0normalization_356/truediv:z:0normalization_357/truediv:z:0normalization_358/truediv:z:0normalization_359/truediv:z:0#concatenate_81/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_81/concat?
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02!
dense_105/MatMul/ReadVariableOp?
dense_105/MatMulMatMulconcatenate_81/concat:output:0'dense_105/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_105/MatMul?
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_105/BiasAdd/ReadVariableOp?
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_105/BiasAddv
dense_105/ReluReludense_105/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_105/Relu?
dense_106/MatMul/ReadVariableOpReadVariableOp(dense_106_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02!
dense_106/MatMul/ReadVariableOp?
dense_106/MatMulMatMuldense_105/Relu:activations:0'dense_106/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_106/MatMul?
 dense_106/BiasAdd/ReadVariableOpReadVariableOp)dense_106_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_106/BiasAdd/ReadVariableOp?
dense_106/BiasAddBiasAdddense_106/MatMul:product:0(dense_106/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_106/BiasAddu
IdentityIdentitydense_106/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp!^dense_106/BiasAdd/ReadVariableOp ^dense_106/MatMul/ReadVariableOp^embedding_239/embedding_lookup^embedding_241/embedding_lookup7^integer_lookup_131/hash_table_Lookup/LookupTableFindV25^string_lookup_87/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2D
 dense_106/BiasAdd/ReadVariableOp dense_106/BiasAdd/ReadVariableOp2B
dense_106/MatMul/ReadVariableOpdense_106/MatMul/ReadVariableOp2@
embedding_239/embedding_lookupembedding_239/embedding_lookup2@
embedding_241/embedding_lookupembedding_241/embedding_lookup2p
6integer_lookup_131/hash_table_Lookup/LookupTableFindV26integer_lookup_131/hash_table_Lookup/LookupTableFindV22l
4string_lookup_87/hash_table_Lookup/LookupTableFindV24string_lookup_87/hash_table_Lookup/LookupTableFindV2:Q M
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
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

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

:
?
H
,__inference_flatten_48_layer_call_fn_2857664

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
G__inference_flatten_48_layer_call_and_return_conditional_losses_28566372
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
??
?
"__inference__wrapped_model_2856574	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_yearsN
Jmodel_52_string_lookup_87_hash_table_lookup_lookuptablefindv2_table_handleO
Kmodel_52_string_lookup_87_hash_table_lookup_lookuptablefindv2_default_value	P
Lmodel_52_integer_lookup_131_hash_table_lookup_lookuptablefindv2_table_handleQ
Mmodel_52_integer_lookup_131_hash_table_lookup_lookuptablefindv2_default_value	A
/model_52_embedding_241_embedding_lookup_2856502:A
/model_52_embedding_239_embedding_lookup_2856507:$
 model_52_normalization_354_sub_y%
!model_52_normalization_354_sqrt_x$
 model_52_normalization_355_sub_y%
!model_52_normalization_355_sqrt_x$
 model_52_normalization_356_sub_y%
!model_52_normalization_356_sqrt_x$
 model_52_normalization_357_sub_y%
!model_52_normalization_357_sqrt_x$
 model_52_normalization_358_sub_y%
!model_52_normalization_358_sqrt_x$
 model_52_normalization_359_sub_y%
!model_52_normalization_359_sqrt_xC
1model_52_dense_105_matmul_readvariableop_resource:
@
2model_52_dense_105_biasadd_readvariableop_resource:
C
1model_52_dense_106_matmul_readvariableop_resource:
@
2model_52_dense_106_biasadd_readvariableop_resource:
identity??)model_52/dense_105/BiasAdd/ReadVariableOp?(model_52/dense_105/MatMul/ReadVariableOp?)model_52/dense_106/BiasAdd/ReadVariableOp?(model_52/dense_106/MatMul/ReadVariableOp?'model_52/embedding_239/embedding_lookup?'model_52/embedding_241/embedding_lookup??model_52/integer_lookup_131/hash_table_Lookup/LookupTableFindV2?=model_52/string_lookup_87/hash_table_Lookup/LookupTableFindV2?
=model_52/string_lookup_87/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Jmodel_52_string_lookup_87_hash_table_lookup_lookuptablefindv2_table_handlestorey_rangeKmodel_52_string_lookup_87_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2?
=model_52/string_lookup_87/hash_table_Lookup/LookupTableFindV2?
"model_52/string_lookup_87/IdentityIdentityFmodel_52/string_lookup_87/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2$
"model_52/string_lookup_87/Identity?
?model_52/integer_lookup_131/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Lmodel_52_integer_lookup_131_hash_table_lookup_lookuptablefindv2_table_handlemonthMmodel_52_integer_lookup_131_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????2A
?model_52/integer_lookup_131/hash_table_Lookup/LookupTableFindV2?
$model_52/integer_lookup_131/IdentityIdentityHmodel_52/integer_lookup_131/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2&
$model_52/integer_lookup_131/Identity?
'model_52/embedding_241/embedding_lookupResourceGather/model_52_embedding_241_embedding_lookup_2856502+model_52/string_lookup_87/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*B
_class8
64loc:@model_52/embedding_241/embedding_lookup/2856502*+
_output_shapes
:?????????*
dtype02)
'model_52/embedding_241/embedding_lookup?
0model_52/embedding_241/embedding_lookup/IdentityIdentity0model_52/embedding_241/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@model_52/embedding_241/embedding_lookup/2856502*+
_output_shapes
:?????????22
0model_52/embedding_241/embedding_lookup/Identity?
2model_52/embedding_241/embedding_lookup/Identity_1Identity9model_52/embedding_241/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????24
2model_52/embedding_241/embedding_lookup/Identity_1?
'model_52/embedding_239/embedding_lookupResourceGather/model_52_embedding_239_embedding_lookup_2856507-model_52/integer_lookup_131/Identity:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0	*B
_class8
64loc:@model_52/embedding_239/embedding_lookup/2856507*+
_output_shapes
:?????????*
dtype02)
'model_52/embedding_239/embedding_lookup?
0model_52/embedding_239/embedding_lookup/IdentityIdentity0model_52/embedding_239/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@model_52/embedding_239/embedding_lookup/2856507*+
_output_shapes
:?????????22
0model_52/embedding_239/embedding_lookup/Identity?
2model_52/embedding_239/embedding_lookup/Identity_1Identity9model_52/embedding_239/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????24
2model_52/embedding_239/embedding_lookup/Identity_1?
model_52/flatten_48/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_52/flatten_48/Const?
model_52/flatten_48/ReshapeReshape;model_52/embedding_239/embedding_lookup/Identity_1:output:0"model_52/flatten_48/Const:output:0*
T0*'
_output_shapes
:?????????2
model_52/flatten_48/Reshape?
model_52/flatten_50/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_52/flatten_50/Const?
model_52/flatten_50/ReshapeReshape;model_52/embedding_241/embedding_lookup/Identity_1:output:0"model_52/flatten_50/Const:output:0*
T0*'
_output_shapes
:?????????2
model_52/flatten_50/Reshape?
model_52/normalization_354/subSubfloor_area_sqm model_52_normalization_354_sub_y*
T0*'
_output_shapes
:?????????2 
model_52/normalization_354/sub?
model_52/normalization_354/SqrtSqrt!model_52_normalization_354_sqrt_x*
T0*
_output_shapes

:2!
model_52/normalization_354/Sqrt?
$model_52/normalization_354/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32&
$model_52/normalization_354/Maximum/y?
"model_52/normalization_354/MaximumMaximum#model_52/normalization_354/Sqrt:y:0-model_52/normalization_354/Maximum/y:output:0*
T0*
_output_shapes

:2$
"model_52/normalization_354/Maximum?
"model_52/normalization_354/truedivRealDiv"model_52/normalization_354/sub:z:0&model_52/normalization_354/Maximum:z:0*
T0*'
_output_shapes
:?????????2$
"model_52/normalization_354/truediv?
model_52/normalization_355/subSubdist_to_nearest_stn model_52_normalization_355_sub_y*
T0*'
_output_shapes
:?????????2 
model_52/normalization_355/sub?
model_52/normalization_355/SqrtSqrt!model_52_normalization_355_sqrt_x*
T0*
_output_shapes

:2!
model_52/normalization_355/Sqrt?
$model_52/normalization_355/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32&
$model_52/normalization_355/Maximum/y?
"model_52/normalization_355/MaximumMaximum#model_52/normalization_355/Sqrt:y:0-model_52/normalization_355/Maximum/y:output:0*
T0*
_output_shapes

:2$
"model_52/normalization_355/Maximum?
"model_52/normalization_355/truedivRealDiv"model_52/normalization_355/sub:z:0&model_52/normalization_355/Maximum:z:0*
T0*'
_output_shapes
:?????????2$
"model_52/normalization_355/truediv?
model_52/normalization_356/subSubdist_to_dhoby model_52_normalization_356_sub_y*
T0*'
_output_shapes
:?????????2 
model_52/normalization_356/sub?
model_52/normalization_356/SqrtSqrt!model_52_normalization_356_sqrt_x*
T0*
_output_shapes

:2!
model_52/normalization_356/Sqrt?
$model_52/normalization_356/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32&
$model_52/normalization_356/Maximum/y?
"model_52/normalization_356/MaximumMaximum#model_52/normalization_356/Sqrt:y:0-model_52/normalization_356/Maximum/y:output:0*
T0*
_output_shapes

:2$
"model_52/normalization_356/Maximum?
"model_52/normalization_356/truedivRealDiv"model_52/normalization_356/sub:z:0&model_52/normalization_356/Maximum:z:0*
T0*'
_output_shapes
:?????????2$
"model_52/normalization_356/truediv?
model_52/normalization_357/subSubdegree_centrality model_52_normalization_357_sub_y*
T0*'
_output_shapes
:?????????2 
model_52/normalization_357/sub?
model_52/normalization_357/SqrtSqrt!model_52_normalization_357_sqrt_x*
T0*
_output_shapes

:2!
model_52/normalization_357/Sqrt?
$model_52/normalization_357/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32&
$model_52/normalization_357/Maximum/y?
"model_52/normalization_357/MaximumMaximum#model_52/normalization_357/Sqrt:y:0-model_52/normalization_357/Maximum/y:output:0*
T0*
_output_shapes

:2$
"model_52/normalization_357/Maximum?
"model_52/normalization_357/truedivRealDiv"model_52/normalization_357/sub:z:0&model_52/normalization_357/Maximum:z:0*
T0*'
_output_shapes
:?????????2$
"model_52/normalization_357/truediv?
model_52/normalization_358/subSubeigenvector_centrality model_52_normalization_358_sub_y*
T0*'
_output_shapes
:?????????2 
model_52/normalization_358/sub?
model_52/normalization_358/SqrtSqrt!model_52_normalization_358_sqrt_x*
T0*
_output_shapes

:2!
model_52/normalization_358/Sqrt?
$model_52/normalization_358/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32&
$model_52/normalization_358/Maximum/y?
"model_52/normalization_358/MaximumMaximum#model_52/normalization_358/Sqrt:y:0-model_52/normalization_358/Maximum/y:output:0*
T0*
_output_shapes

:2$
"model_52/normalization_358/Maximum?
"model_52/normalization_358/truedivRealDiv"model_52/normalization_358/sub:z:0&model_52/normalization_358/Maximum:z:0*
T0*'
_output_shapes
:?????????2$
"model_52/normalization_358/truediv?
model_52/normalization_359/subSubremaining_lease_years model_52_normalization_359_sub_y*
T0*'
_output_shapes
:?????????2 
model_52/normalization_359/sub?
model_52/normalization_359/SqrtSqrt!model_52_normalization_359_sqrt_x*
T0*
_output_shapes

:2!
model_52/normalization_359/Sqrt?
$model_52/normalization_359/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32&
$model_52/normalization_359/Maximum/y?
"model_52/normalization_359/MaximumMaximum#model_52/normalization_359/Sqrt:y:0-model_52/normalization_359/Maximum/y:output:0*
T0*
_output_shapes

:2$
"model_52/normalization_359/Maximum?
"model_52/normalization_359/truedivRealDiv"model_52/normalization_359/sub:z:0&model_52/normalization_359/Maximum:z:0*
T0*'
_output_shapes
:?????????2$
"model_52/normalization_359/truediv?
#model_52/concatenate_81/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_52/concatenate_81/concat/axis?
model_52/concatenate_81/concatConcatV2$model_52/flatten_48/Reshape:output:0$model_52/flatten_50/Reshape:output:0&model_52/normalization_354/truediv:z:0&model_52/normalization_355/truediv:z:0&model_52/normalization_356/truediv:z:0&model_52/normalization_357/truediv:z:0&model_52/normalization_358/truediv:z:0&model_52/normalization_359/truediv:z:0,model_52/concatenate_81/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2 
model_52/concatenate_81/concat?
(model_52/dense_105/MatMul/ReadVariableOpReadVariableOp1model_52_dense_105_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02*
(model_52/dense_105/MatMul/ReadVariableOp?
model_52/dense_105/MatMulMatMul'model_52/concatenate_81/concat:output:00model_52/dense_105/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model_52/dense_105/MatMul?
)model_52/dense_105/BiasAdd/ReadVariableOpReadVariableOp2model_52_dense_105_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)model_52/dense_105/BiasAdd/ReadVariableOp?
model_52/dense_105/BiasAddBiasAdd#model_52/dense_105/MatMul:product:01model_52/dense_105/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model_52/dense_105/BiasAdd?
model_52/dense_105/ReluRelu#model_52/dense_105/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
model_52/dense_105/Relu?
(model_52/dense_106/MatMul/ReadVariableOpReadVariableOp1model_52_dense_106_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02*
(model_52/dense_106/MatMul/ReadVariableOp?
model_52/dense_106/MatMulMatMul%model_52/dense_105/Relu:activations:00model_52/dense_106/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_52/dense_106/MatMul?
)model_52/dense_106/BiasAdd/ReadVariableOpReadVariableOp2model_52_dense_106_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_52/dense_106/BiasAdd/ReadVariableOp?
model_52/dense_106/BiasAddBiasAdd#model_52/dense_106/MatMul:product:01model_52/dense_106/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_52/dense_106/BiasAdd~
IdentityIdentity#model_52/dense_106/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp*^model_52/dense_105/BiasAdd/ReadVariableOp)^model_52/dense_105/MatMul/ReadVariableOp*^model_52/dense_106/BiasAdd/ReadVariableOp)^model_52/dense_106/MatMul/ReadVariableOp(^model_52/embedding_239/embedding_lookup(^model_52/embedding_241/embedding_lookup@^model_52/integer_lookup_131/hash_table_Lookup/LookupTableFindV2>^model_52/string_lookup_87/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2V
)model_52/dense_105/BiasAdd/ReadVariableOp)model_52/dense_105/BiasAdd/ReadVariableOp2T
(model_52/dense_105/MatMul/ReadVariableOp(model_52/dense_105/MatMul/ReadVariableOp2V
)model_52/dense_106/BiasAdd/ReadVariableOp)model_52/dense_106/BiasAdd/ReadVariableOp2T
(model_52/dense_106/MatMul/ReadVariableOp(model_52/dense_106/MatMul/ReadVariableOp2R
'model_52/embedding_239/embedding_lookup'model_52/embedding_239/embedding_lookup2R
'model_52/embedding_241/embedding_lookup'model_52/embedding_241/embedding_lookup2?
?model_52/integer_lookup_131/hash_table_Lookup/LookupTableFindV2?model_52/integer_lookup_131/hash_table_Lookup/LookupTableFindV22~
=model_52/string_lookup_87/hash_table_Lookup/LookupTableFindV2=model_52/string_lookup_87/hash_table_Lookup/LookupTableFindV2:N J
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
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

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

:
?,
?
__inference_adapt_step_2083388
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
F__inference_dense_105_layer_call_and_return_conditional_losses_2857711

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_dense_105_layer_call_and_return_conditional_losses_2856715

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
__inference__creator_2857762
identity:	 ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_nametable_2075580*
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
?e
?
E__inference_model_52_layer_call_and_return_conditional_losses_2857179	
month	
flat_model_type
storey_range
floor_area_sqm
dist_to_nearest_stn
dist_to_dhoby
degree_centrality
eigenvector_centrality
remaining_lease_yearsE
Astring_lookup_87_hash_table_lookup_lookuptablefindv2_table_handleF
Bstring_lookup_87_hash_table_lookup_lookuptablefindv2_default_value	G
Cinteger_lookup_131_hash_table_lookup_lookuptablefindv2_table_handleH
Dinteger_lookup_131_hash_table_lookup_lookuptablefindv2_default_value	'
embedding_241_2857117:'
embedding_239_2857120:
normalization_354_sub_y
normalization_354_sqrt_x
normalization_355_sub_y
normalization_355_sqrt_x
normalization_356_sub_y
normalization_356_sqrt_x
normalization_357_sub_y
normalization_357_sqrt_x
normalization_358_sub_y
normalization_358_sqrt_x
normalization_359_sub_y
normalization_359_sqrt_x#
dense_105_2857168:

dense_105_2857170:
#
dense_106_2857173:

dense_106_2857175:
identity??!dense_105/StatefulPartitionedCall?!dense_106/StatefulPartitionedCall?%embedding_239/StatefulPartitionedCall?%embedding_241/StatefulPartitionedCall?6integer_lookup_131/hash_table_Lookup/LookupTableFindV2?4string_lookup_87/hash_table_Lookup/LookupTableFindV2?
4string_lookup_87/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Astring_lookup_87_hash_table_lookup_lookuptablefindv2_table_handlestorey_rangeBstring_lookup_87_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????26
4string_lookup_87/hash_table_Lookup/LookupTableFindV2?
string_lookup_87/IdentityIdentity=string_lookup_87/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_87/Identity?
6integer_lookup_131/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Cinteger_lookup_131_hash_table_lookup_lookuptablefindv2_table_handlemonthDinteger_lookup_131_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*'
_output_shapes
:?????????28
6integer_lookup_131/hash_table_Lookup/LookupTableFindV2?
integer_lookup_131/IdentityIdentity?integer_lookup_131/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
integer_lookup_131/Identity?
%embedding_241/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_87/Identity:output:0embedding_241_2857117*
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
J__inference_embedding_241_layer_call_and_return_conditional_losses_28566142'
%embedding_241/StatefulPartitionedCall?
%embedding_239/StatefulPartitionedCallStatefulPartitionedCall$integer_lookup_131/Identity:output:0embedding_239_2857120*
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
J__inference_embedding_239_layer_call_and_return_conditional_losses_28566272'
%embedding_239/StatefulPartitionedCall?
flatten_48/PartitionedCallPartitionedCall.embedding_239/StatefulPartitionedCall:output:0*
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
G__inference_flatten_48_layer_call_and_return_conditional_losses_28566372
flatten_48/PartitionedCall?
flatten_50/PartitionedCallPartitionedCall.embedding_241/StatefulPartitionedCall:output:0*
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
G__inference_flatten_50_layer_call_and_return_conditional_losses_28566452
flatten_50/PartitionedCall?
normalization_354/subSubfloor_area_sqmnormalization_354_sub_y*
T0*'
_output_shapes
:?????????2
normalization_354/sub{
normalization_354/SqrtSqrtnormalization_354_sqrt_x*
T0*
_output_shapes

:2
normalization_354/Sqrt
normalization_354/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_354/Maximum/y?
normalization_354/MaximumMaximumnormalization_354/Sqrt:y:0$normalization_354/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_354/Maximum?
normalization_354/truedivRealDivnormalization_354/sub:z:0normalization_354/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_354/truediv?
normalization_355/subSubdist_to_nearest_stnnormalization_355_sub_y*
T0*'
_output_shapes
:?????????2
normalization_355/sub{
normalization_355/SqrtSqrtnormalization_355_sqrt_x*
T0*
_output_shapes

:2
normalization_355/Sqrt
normalization_355/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_355/Maximum/y?
normalization_355/MaximumMaximumnormalization_355/Sqrt:y:0$normalization_355/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_355/Maximum?
normalization_355/truedivRealDivnormalization_355/sub:z:0normalization_355/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_355/truediv?
normalization_356/subSubdist_to_dhobynormalization_356_sub_y*
T0*'
_output_shapes
:?????????2
normalization_356/sub{
normalization_356/SqrtSqrtnormalization_356_sqrt_x*
T0*
_output_shapes

:2
normalization_356/Sqrt
normalization_356/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_356/Maximum/y?
normalization_356/MaximumMaximumnormalization_356/Sqrt:y:0$normalization_356/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_356/Maximum?
normalization_356/truedivRealDivnormalization_356/sub:z:0normalization_356/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_356/truediv?
normalization_357/subSubdegree_centralitynormalization_357_sub_y*
T0*'
_output_shapes
:?????????2
normalization_357/sub{
normalization_357/SqrtSqrtnormalization_357_sqrt_x*
T0*
_output_shapes

:2
normalization_357/Sqrt
normalization_357/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_357/Maximum/y?
normalization_357/MaximumMaximumnormalization_357/Sqrt:y:0$normalization_357/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_357/Maximum?
normalization_357/truedivRealDivnormalization_357/sub:z:0normalization_357/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_357/truediv?
normalization_358/subSubeigenvector_centralitynormalization_358_sub_y*
T0*'
_output_shapes
:?????????2
normalization_358/sub{
normalization_358/SqrtSqrtnormalization_358_sqrt_x*
T0*
_output_shapes

:2
normalization_358/Sqrt
normalization_358/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_358/Maximum/y?
normalization_358/MaximumMaximumnormalization_358/Sqrt:y:0$normalization_358/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_358/Maximum?
normalization_358/truedivRealDivnormalization_358/sub:z:0normalization_358/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_358/truediv?
normalization_359/subSubremaining_lease_yearsnormalization_359_sub_y*
T0*'
_output_shapes
:?????????2
normalization_359/sub{
normalization_359/SqrtSqrtnormalization_359_sqrt_x*
T0*
_output_shapes

:2
normalization_359/Sqrt
normalization_359/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_359/Maximum/y?
normalization_359/MaximumMaximumnormalization_359/Sqrt:y:0$normalization_359/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_359/Maximum?
normalization_359/truedivRealDivnormalization_359/sub:z:0normalization_359/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_359/truediv?
concatenate_81/PartitionedCallPartitionedCall#flatten_48/PartitionedCall:output:0#flatten_50/PartitionedCall:output:0normalization_354/truediv:z:0normalization_355/truediv:z:0normalization_356/truediv:z:0normalization_357/truediv:z:0normalization_358/truediv:z:0normalization_359/truediv:z:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_concatenate_81_layer_call_and_return_conditional_losses_28567022 
concatenate_81/PartitionedCall?
!dense_105/StatefulPartitionedCallStatefulPartitionedCall'concatenate_81/PartitionedCall:output:0dense_105_2857168dense_105_2857170*
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
GPU2*0J 8? *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_28567152#
!dense_105/StatefulPartitionedCall?
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0dense_106_2857173dense_106_2857175*
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
GPU2*0J 8? *O
fJRH
F__inference_dense_106_layer_call_and_return_conditional_losses_28567312#
!dense_106/StatefulPartitionedCall?
IdentityIdentity*dense_106/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall&^embedding_239/StatefulPartitionedCall&^embedding_241/StatefulPartitionedCall7^integer_lookup_131/hash_table_Lookup/LookupTableFindV25^string_lookup_87/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : ::::::::::::: : : : 2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2N
%embedding_239/StatefulPartitionedCall%embedding_239/StatefulPartitionedCall2N
%embedding_241/StatefulPartitionedCall%embedding_241/StatefulPartitionedCall2p
6integer_lookup_131/hash_table_Lookup/LookupTableFindV26integer_lookup_131/hash_table_Lookup/LookupTableFindV22l
4string_lookup_87/hash_table_Lookup/LookupTableFindV24string_lookup_87/hash_table_Lookup/LookupTableFindV2:N J
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
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

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

:
?
0
 __inference__initializer_2857800
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
??
?
#__inference__traced_restore_2858262
file_prefixM
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable:	 Q
Gmutablehashtable_table_restore_1_lookuptableimportv2_mutablehashtable_1: ;
)assignvariableop_embedding_239_embeddings:=
+assignvariableop_1_embedding_241_embeddings:%
assignvariableop_2_mean:)
assignvariableop_3_variance:"
assignvariableop_4_count:	 '
assignvariableop_5_mean_1:+
assignvariableop_6_variance_1:$
assignvariableop_7_count_1:	 '
assignvariableop_8_mean_2:+
assignvariableop_9_variance_2:%
assignvariableop_10_count_2:	 (
assignvariableop_11_mean_3:,
assignvariableop_12_variance_3:%
assignvariableop_13_count_3:	 (
assignvariableop_14_mean_4:,
assignvariableop_15_variance_4:%
assignvariableop_16_count_4:	 (
assignvariableop_17_mean_5:,
assignvariableop_18_variance_5:%
assignvariableop_19_count_5:	 6
$assignvariableop_20_dense_105_kernel:
0
"assignvariableop_21_dense_105_bias:
6
$assignvariableop_22_dense_106_kernel:
0
"assignvariableop_23_dense_106_bias:'
assignvariableop_24_adam_iter:	 )
assignvariableop_25_adam_beta_1: )
assignvariableop_26_adam_beta_2: (
assignvariableop_27_adam_decay: 0
&assignvariableop_28_adam_learning_rate: #
assignvariableop_29_total: %
assignvariableop_30_count_6: %
assignvariableop_31_total_1: %
assignvariableop_32_count_7: %
assignvariableop_33_total_2: %
assignvariableop_34_count_8: %
assignvariableop_35_total_3: %
assignvariableop_36_count_9: E
3assignvariableop_37_adam_embedding_239_embeddings_m:E
3assignvariableop_38_adam_embedding_241_embeddings_m:=
+assignvariableop_39_adam_dense_105_kernel_m:
7
)assignvariableop_40_adam_dense_105_bias_m:
=
+assignvariableop_41_adam_dense_106_kernel_m:
7
)assignvariableop_42_adam_dense_106_bias_m:E
3assignvariableop_43_adam_embedding_239_embeddings_v:E
3assignvariableop_44_adam_embedding_241_embeddings_v:=
+assignvariableop_45_adam_dense_105_kernel_v:
7
)assignvariableop_46_adam_dense_105_bias_v:
=
+assignvariableop_47_adam_dense_106_kernel_v:
7
)assignvariableop_48_adam_dense_106_bias_v:
identity_50??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?4MutableHashTable_table_restore_1/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*?
value?B?6B8layer_with_weights-0/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-0/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-1/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-1/token_counts/.ATTRIBUTES/table-valuesB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-7/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-8/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-9/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/count/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826										2
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
4MutableHashTable_table_restore_1/LookupTableImportV2g
IdentityIdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp)assignvariableop_embedding_239_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_embedding_241_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_countIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_mean_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5l

Identity_6IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_variance_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6l

Identity_7IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_count_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_7l

Identity_8IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_mean_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8l

Identity_9IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_variance_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_count_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_mean_3Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_variance_3Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_3Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_mean_4Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_variance_4Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_4Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_mean_5Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_variance_5Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_5Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp$assignvariableop_20_dense_105_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_105_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp$assignvariableop_22_dense_106_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp"assignvariableop_23_dense_106_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_6Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_7Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_2Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_8Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_3Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_9Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp3assignvariableop_37_adam_embedding_239_embeddings_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_embedding_241_embeddings_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_dense_105_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_105_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_dense_106_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_106_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp3assignvariableop_43_adam_embedding_239_embeddings_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp3assignvariableop_44_adam_embedding_241_embeddings_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_dense_105_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_105_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_dense_106_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_106_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV25^MutableHashTable_table_restore_1/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49f
Identity_50IdentityIdentity_49:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_50?	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV25^MutableHashTable_table_restore_1/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_50Identity_50:output:0*{
_input_shapesj
h: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV22l
4MutableHashTable_table_restore_1/LookupTableImportV24MutableHashTable_table_restore_1/LookupTableImportV2:C ?
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
loc:@MutableHashTable_1
?d
?
 __inference__traced_save_2858099
file_prefixJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2	L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	L
Hsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2N
Jsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2_1	7
3savev2_embedding_239_embeddings_read_readvariableop7
3savev2_embedding_241_embeddings_read_readvariableop#
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
"savev2_count_5_read_readvariableop	/
+savev2_dense_105_kernel_read_readvariableop-
)savev2_dense_105_bias_read_readvariableop/
+savev2_dense_106_kernel_read_readvariableop-
)savev2_dense_106_bias_read_readvariableop(
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
:savev2_adam_embedding_239_embeddings_m_read_readvariableop>
:savev2_adam_embedding_241_embeddings_m_read_readvariableop6
2savev2_adam_dense_105_kernel_m_read_readvariableop4
0savev2_adam_dense_105_bias_m_read_readvariableop6
2savev2_adam_dense_106_kernel_m_read_readvariableop4
0savev2_adam_dense_106_bias_m_read_readvariableop>
:savev2_adam_embedding_239_embeddings_v_read_readvariableop>
:savev2_adam_embedding_241_embeddings_v_read_readvariableop6
2savev2_adam_dense_105_kernel_v_read_readvariableop4
0savev2_adam_dense_105_bias_v_read_readvariableop6
2savev2_adam_dense_106_kernel_v_read_readvariableop4
0savev2_adam_dense_106_bias_v_read_readvariableop
savev2_const_20

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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*?
value?B?6B8layer_with_weights-0/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-0/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-1/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-1/token_counts/.ATTRIBUTES/table-valuesB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-7/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-8/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-9/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/count/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1Hsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2Jsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2_13savev2_embedding_239_embeddings_read_readvariableop3savev2_embedding_241_embeddings_read_readvariableopsavev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop!savev2_mean_1_read_readvariableop%savev2_variance_1_read_readvariableop"savev2_count_1_read_readvariableop!savev2_mean_2_read_readvariableop%savev2_variance_2_read_readvariableop"savev2_count_2_read_readvariableop!savev2_mean_3_read_readvariableop%savev2_variance_3_read_readvariableop"savev2_count_3_read_readvariableop!savev2_mean_4_read_readvariableop%savev2_variance_4_read_readvariableop"savev2_count_4_read_readvariableop!savev2_mean_5_read_readvariableop%savev2_variance_5_read_readvariableop"savev2_count_5_read_readvariableop+savev2_dense_105_kernel_read_readvariableop)savev2_dense_105_bias_read_readvariableop+savev2_dense_106_kernel_read_readvariableop)savev2_dense_106_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop"savev2_count_6_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_7_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_8_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_9_read_readvariableop:savev2_adam_embedding_239_embeddings_m_read_readvariableop:savev2_adam_embedding_241_embeddings_m_read_readvariableop2savev2_adam_dense_105_kernel_m_read_readvariableop0savev2_adam_dense_105_bias_m_read_readvariableop2savev2_adam_dense_106_kernel_m_read_readvariableop0savev2_adam_dense_106_bias_m_read_readvariableop:savev2_adam_embedding_239_embeddings_v_read_readvariableop:savev2_adam_embedding_241_embeddings_v_read_readvariableop2savev2_adam_dense_105_kernel_v_read_readvariableop0savev2_adam_dense_105_bias_v_read_readvariableop2savev2_adam_dense_106_kernel_v_read_readvariableop0savev2_adam_dense_106_bias_v_read_readvariableopsavev2_const_20"/device:CPU:0*
_output_shapes
 *D
dtypes:
826										2
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
?: ::::::::: ::: ::: ::: ::: ::: :
:
:
:: : : : : : : : : : : : : :::
:
:
::::
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
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::	

_output_shapes
: : 
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
: :$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 
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
: :$* 

_output_shapes

::$+ 

_output_shapes

::$, 

_output_shapes

:
: -

_output_shapes
:
:$. 

_output_shapes

:
: /

_output_shapes
::$0 

_output_shapes

::$1 

_output_shapes

::$2 

_output_shapes

:
: 3

_output_shapes
:
:$4 

_output_shapes

:
: 5

_output_shapes
::6

_output_shapes
: 
?
.
__inference__destroyer_2857790
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
0__inference_concatenate_81_layer_call_fn_2857700
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_concatenate_81_layer_call_and_return_conditional_losses_28567022
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
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
inputs/7
?
?
+__inference_dense_105_layer_call_fn_2857720

inputs
unknown:
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
GPU2*0J 8? *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_28567152
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
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_3:0StatefulPartitionedCall_48"
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
serving_default_storey_range:0??????????
	dense_1062
StatefulPartitionedCall_2:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
layer_with_weights-5
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer_with_weights-8
layer-18
layer_with_weights-9
layer-19
layer-20
layer_with_weights-10
layer-21
layer-22
layer_with_weights-11
layer-23
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
b
lookup_table
 token_counts
!	keras_api
?_adapt_function"
_tf_keras_layer
b
"lookup_table
#token_counts
$	keras_api
?_adapt_function"
_tf_keras_layer
?
%
embeddings
&regularization_losses
'	variables
(trainable_variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
*
embeddings
+regularization_losses
,	variables
-trainable_variables
.	keras_api
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
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
7
_keep_axis
8_reduce_axis
9_reduce_axis_mask
:_broadcast_shape
;mean
;
adapt_mean
<variance
<adapt_variance
	=count
>	keras_api
?_adapt_function"
_tf_keras_layer
?
?
_keep_axis
@_reduce_axis
A_reduce_axis_mask
B_broadcast_shape
Cmean
C
adapt_mean
Dvariance
Dadapt_variance
	Ecount
F	keras_api
?_adapt_function"
_tf_keras_layer
?
G
_keep_axis
H_reduce_axis
I_reduce_axis_mask
J_broadcast_shape
Kmean
K
adapt_mean
Lvariance
Ladapt_variance
	Mcount
N	keras_api
?_adapt_function"
_tf_keras_layer
?
O
_keep_axis
P_reduce_axis
Q_reduce_axis_mask
R_broadcast_shape
Smean
S
adapt_mean
Tvariance
Tadapt_variance
	Ucount
V	keras_api
?_adapt_function"
_tf_keras_layer
?
W
_keep_axis
X_reduce_axis
Y_reduce_axis_mask
Z_broadcast_shape
[mean
[
adapt_mean
\variance
\adapt_variance
	]count
^	keras_api
?_adapt_function"
_tf_keras_layer
?
_
_keep_axis
`_reduce_axis
a_reduce_axis_mask
b_broadcast_shape
cmean
c
adapt_mean
dvariance
dadapt_variance
	ecount
f	keras_api
?_adapt_function"
_tf_keras_layer
?
gregularization_losses
h	variables
itrainable_variables
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

kkernel
lbias
mregularization_losses
n	variables
otrainable_variables
p	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
"
_tf_keras_input_layer
?

qkernel
rbias
sregularization_losses
t	variables
utrainable_variables
v	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
witer

xbeta_1

ybeta_2
	zdecay
{learning_rate%m?*m?km?lm?qm?rm?%v?*v?kv?lv?qv?rv?"
	optimizer
 "
trackable_list_wrapper
?
%2
*3
;4
<5
=6
C7
D8
E9
K10
L11
M12
S13
T14
U15
[16
\17
]18
c19
d20
e21
k22
l23
q24
r25"
trackable_list_wrapper
J
%0
*1
k2
l3
q4
r5"
trackable_list_wrapper
?
|layer_metrics

}layers
regularization_losses
~non_trainable_variables
metrics
	variables
 ?layer_regularization_losses
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
*:(2embedding_239/embeddings
 "
trackable_list_wrapper
'
%0"
trackable_list_wrapper
'
%0"
trackable_list_wrapper
?
?layer_metrics
?layers
&regularization_losses
?non_trainable_variables
?metrics
'	variables
 ?layer_regularization_losses
(trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2embedding_241/embeddings
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
?
?layer_metrics
?layers
+regularization_losses
?non_trainable_variables
?metrics
,	variables
 ?layer_regularization_losses
-trainable_variables
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
/regularization_losses
?non_trainable_variables
?metrics
0	variables
 ?layer_regularization_losses
1trainable_variables
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
3regularization_losses
?non_trainable_variables
?metrics
4	variables
 ?layer_regularization_losses
5trainable_variables
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
gregularization_losses
?non_trainable_variables
?metrics
h	variables
 ?layer_regularization_losses
itrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_105/kernel
:
2dense_105/bias
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
?
?layer_metrics
?layers
mregularization_losses
?non_trainable_variables
?metrics
n	variables
 ?layer_regularization_losses
otrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_106/kernel
:2dense_106/bias
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
?
?layer_metrics
?layers
sregularization_losses
?non_trainable_variables
?metrics
t	variables
 ?layer_regularization_losses
utrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
23"
trackable_list_wrapper
?
;2
<3
=4
C5
D6
E7
K8
L9
M10
S11
T12
U13
[14
\15
]16
c17
d18
e19"
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
/:-2Adam/embedding_239/embeddings/m
/:-2Adam/embedding_241/embeddings/m
':%
2Adam/dense_105/kernel/m
!:
2Adam/dense_105/bias/m
':%
2Adam/dense_106/kernel/m
!:2Adam/dense_106/bias/m
/:-2Adam/embedding_239/embeddings/v
/:-2Adam/embedding_241/embeddings/v
':%
2Adam/dense_105/kernel/v
!:
2Adam/dense_105/bias/v
':%
2Adam/dense_106/kernel/v
!:2Adam/dense_106/bias/v
?2?
E__inference_model_52_layer_call_and_return_conditional_losses_2857416
E__inference_model_52_layer_call_and_return_conditional_losses_2857507
E__inference_model_52_layer_call_and_return_conditional_losses_2857179
E__inference_model_52_layer_call_and_return_conditional_losses_2857260?
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
"__inference__wrapped_model_2856574monthflat_model_typestorey_rangefloor_area_sqmdist_to_nearest_stndist_to_dhobydegree_centralityeigenvector_centralityremaining_lease_years	"?
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
*__inference_model_52_layer_call_fn_2856785
*__inference_model_52_layer_call_fn_2857564
*__inference_model_52_layer_call_fn_2857621
*__inference_model_52_layer_call_fn_2857098?
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
__inference_adapt_step_2488642?
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
__inference_adapt_step_2083099?
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
J__inference_embedding_239_layer_call_and_return_conditional_losses_2857630?
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
/__inference_embedding_239_layer_call_fn_2857637?
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
J__inference_embedding_241_layer_call_and_return_conditional_losses_2857646?
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
/__inference_embedding_241_layer_call_fn_2857653?
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
G__inference_flatten_48_layer_call_and_return_conditional_losses_2857659?
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
,__inference_flatten_48_layer_call_fn_2857664?
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
G__inference_flatten_50_layer_call_and_return_conditional_losses_2857670?
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
,__inference_flatten_50_layer_call_fn_2857675?
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
__inference_adapt_step_2083200?
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
__inference_adapt_step_2083247?
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
__inference_adapt_step_2083294?
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
__inference_adapt_step_2083341?
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
__inference_adapt_step_2083388?
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
__inference_adapt_step_2083435?
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
K__inference_concatenate_81_layer_call_and_return_conditional_losses_2857688?
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
0__inference_concatenate_81_layer_call_fn_2857700?
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
F__inference_dense_105_layer_call_and_return_conditional_losses_2857711?
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
+__inference_dense_105_layer_call_fn_2857720?
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
F__inference_dense_106_layer_call_and_return_conditional_losses_2857730?
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
+__inference_dense_106_layer_call_fn_2857739?
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
%__inference_signature_wrapper_2857325degree_centralitydist_to_dhobydist_to_nearest_stneigenvector_centralityflat_model_typefloor_area_sqmmonthremaining_lease_yearsstorey_range"?
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
__inference__creator_2857744?
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
 __inference__initializer_2857752?
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
__inference__destroyer_2857757?
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
__inference__creator_2857762?
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
 __inference__initializer_2857767?
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
__inference__destroyer_2857772?
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
__inference_save_fn_2857824checkpoint_key"?
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
__inference_restore_fn_2857832restored_tensors_0restored_tensors_1"?
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
__inference__creator_2857777?
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
 __inference__initializer_2857785?
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
__inference__destroyer_2857790?
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
__inference__creator_2857795?
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
 __inference__initializer_2857800?
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
__inference__destroyer_2857805?
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
__inference_save_fn_2857851checkpoint_key"?
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
__inference_restore_fn_2857859restored_tensors_0restored_tensors_1"?
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

Const_198
__inference__creator_2857744?

? 
? "? 8
__inference__creator_2857762?

? 
? "? 8
__inference__creator_2857777?

? 
? "? 8
__inference__creator_2857795?

? 
? "? :
__inference__destroyer_2857757?

? 
? "? :
__inference__destroyer_2857772?

? 
? "? :
__inference__destroyer_2857790?

? 
? "? :
__inference__destroyer_2857805?

? 
? "? C
 __inference__initializer_2857752???

? 
? "? <
 __inference__initializer_2857767?

? 
? "? C
 __inference__initializer_2857785"???

? 
? "? <
 __inference__initializer_2857800?

? 
? "? ?
"__inference__wrapped_model_2856574?$"??*%????????????klqr???
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
? "5?2
0
	dense_106#? 
	dense_106?????????n
__inference_adapt_step_2083099L#?A?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_2083200L=;<A?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_2083247LECDA?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_2083294LMKLA?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_2083341LUSTA?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_2083388L][\A?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_2083435LecdA?>
7?4
2?/?
??????????IteratorSpec
? "
 n
__inference_adapt_step_2488642L ?A?>
7?4
2?/?
??????????	IteratorSpec
? "
 ?
K__inference_concatenate_81_layer_call_and_return_conditional_losses_2857688????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
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
? "%?"
?
0?????????
? ?
0__inference_concatenate_81_layer_call_fn_2857700????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
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
? "???????????
F__inference_dense_105_layer_call_and_return_conditional_losses_2857711\kl/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? ~
+__inference_dense_105_layer_call_fn_2857720Okl/?,
%?"
 ?
inputs?????????
? "??????????
?
F__inference_dense_106_layer_call_and_return_conditional_losses_2857730\qr/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? ~
+__inference_dense_106_layer_call_fn_2857739Oqr/?,
%?"
 ?
inputs?????????

? "???????????
J__inference_embedding_239_layer_call_and_return_conditional_losses_2857630_%/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????
? ?
/__inference_embedding_239_layer_call_fn_2857637R%/?,
%?"
 ?
inputs?????????	
? "???????????
J__inference_embedding_241_layer_call_and_return_conditional_losses_2857646_*/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????
? ?
/__inference_embedding_241_layer_call_fn_2857653R*/?,
%?"
 ?
inputs?????????	
? "???????????
G__inference_flatten_48_layer_call_and_return_conditional_losses_2857659\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? 
,__inference_flatten_48_layer_call_fn_2857664O3?0
)?&
$?!
inputs?????????
? "???????????
G__inference_flatten_50_layer_call_and_return_conditional_losses_2857670\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? 
,__inference_flatten_50_layer_call_fn_2857675O3?0
)?&
$?!
inputs?????????
? "???????????
E__inference_model_52_layer_call_and_return_conditional_losses_2857179?$"??*%????????????klqr???
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
E__inference_model_52_layer_call_and_return_conditional_losses_2857260?$"??*%????????????klqr???
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
? ?
E__inference_model_52_layer_call_and_return_conditional_losses_2857416?$"??*%????????????klqr???
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
? ?
E__inference_model_52_layer_call_and_return_conditional_losses_2857507?$"??*%????????????klqr???
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
*__inference_model_52_layer_call_fn_2856785?$"??*%????????????klqr???
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
*__inference_model_52_layer_call_fn_2857098?$"??*%????????????klqr???
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
*__inference_model_52_layer_call_fn_2857564?$"??*%????????????klqr???
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
*__inference_model_52_layer_call_fn_2857621?$"??*%????????????klqr???
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
__inference_restore_fn_2857832Y K?H
A?>
?
restored_tensors_0	
?
restored_tensors_1	
? "? {
__inference_restore_fn_2857859Y#K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_2857824? &?#
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
__inference_save_fn_2857851?#&?#
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
%__inference_signature_wrapper_2857325?$"??*%????????????klqr???
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
storey_range?????????"5?2
0
	dense_106#? 
	dense_106?????????