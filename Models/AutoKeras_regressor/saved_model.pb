др
Щэ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.1.02unknown8Џ™
|
normalization/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namenormalization/mean
u
&normalization/mean/Read/ReadVariableOpReadVariableOpnormalization/mean*
_output_shapes
:*
dtype0
Д
normalization/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namenormalization/variance
}
*normalization/variance/Read/ReadVariableOpReadVariableOpnormalization/variance*
_output_shapes
:*
dtype0
z
normalization/countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namenormalization/count
s
'normalization/count/Read/ReadVariableOpReadVariableOpnormalization/count*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
¶
!separable_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!separable_conv2d/depthwise_kernel
Я
5separable_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/depthwise_kernel*&
_output_shapes
:@*
dtype0
І
!separable_conv2d/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*2
shared_name#!separable_conv2d/pointwise_kernel
†
5separable_conv2d/pointwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/pointwise_kernel*'
_output_shapes
:@А*
dtype0
Г
separable_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameseparable_conv2d/bias
|
)separable_conv2d/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d/bias*
_output_shapes	
:А*
dtype0
Ђ
#separable_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_1/depthwise_kernel
§
7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/depthwise_kernel*'
_output_shapes
:А*
dtype0
ђ
#separable_conv2d_1/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_1/pointwise_kernel
•
7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_1/bias
А
+separable_conv2d_1/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_1/bias*
_output_shapes	
:А*
dtype0
Г
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:@А*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:А*
dtype0
Ђ
#separable_conv2d_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_2/depthwise_kernel
§
7separable_conv2d_2/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_2/depthwise_kernel*'
_output_shapes
:А*
dtype0
ђ
#separable_conv2d_2/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_2/pointwise_kernel
•
7separable_conv2d_2/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_2/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_2/bias
А
+separable_conv2d_2/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_2/bias*
_output_shapes	
:А*
dtype0
Ђ
#separable_conv2d_3/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_3/depthwise_kernel
§
7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/depthwise_kernel*'
_output_shapes
:А*
dtype0
ђ
#separable_conv2d_3/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_3/pointwise_kernel
•
7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_3/bias
А
+separable_conv2d_3/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_3/bias*
_output_shapes	
:А*
dtype0
Ђ
#separable_conv2d_4/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_4/depthwise_kernel
§
7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/depthwise_kernel*'
_output_shapes
:А*
dtype0
ђ
#separable_conv2d_4/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_4/pointwise_kernel
•
7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_4/bias
А
+separable_conv2d_4/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_4/bias*
_output_shapes	
:А*
dtype0
Ђ
#separable_conv2d_5/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#separable_conv2d_5/depthwise_kernel
§
7separable_conv2d_5/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_5/depthwise_kernel*'
_output_shapes
:А*
dtype0
ђ
#separable_conv2d_5/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#separable_conv2d_5/pointwise_kernel
•
7separable_conv2d_5/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_5/pointwise_kernel*(
_output_shapes
:АА*
dtype0
З
separable_conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d_5/bias
А
+separable_conv2d_5/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_5/bias*
_output_shapes	
:А*
dtype0
Д
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:АА*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:А*
dtype0
Н
regression_head_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*)
shared_nameregression_head_1/kernel
Ж
,regression_head_1/kernel/Read/ReadVariableOpReadVariableOpregression_head_1/kernel*
_output_shapes
:	А*
dtype0
Д
regression_head_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameregression_head_1/bias
}
*regression_head_1/bias/Read/ReadVariableOpReadVariableOpregression_head_1/bias*
_output_shapes
:*
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
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
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
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
В
conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d/kernel/m
{
#conv2d/kernel/m/Read/ReadVariableOpReadVariableOpconv2d/kernel/m*&
_output_shapes
:@*
dtype0
r
conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias/m
k
!conv2d/bias/m/Read/ReadVariableOpReadVariableOpconv2d/bias/m*
_output_shapes
:@*
dtype0
™
#separable_conv2d/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#separable_conv2d/depthwise_kernel/m
£
7separable_conv2d/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp#separable_conv2d/depthwise_kernel/m*&
_output_shapes
:@*
dtype0
Ђ
#separable_conv2d/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*4
shared_name%#separable_conv2d/pointwise_kernel/m
§
7separable_conv2d/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp#separable_conv2d/pointwise_kernel/m*'
_output_shapes
:@А*
dtype0
З
separable_conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d/bias/m
А
+separable_conv2d/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d/bias/m*
_output_shapes	
:А*
dtype0
ѓ
%separable_conv2d_1/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_1/depthwise_kernel/m
®
9separable_conv2d_1/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_1/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
∞
%separable_conv2d_1/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_1/pointwise_kernel/m
©
9separable_conv2d_1/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_1/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_1/bias/m
Д
-separable_conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_1/bias/m*
_output_shapes	
:А*
dtype0
З
conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*"
shared_nameconv2d_1/kernel/m
А
%conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_1/kernel/m*'
_output_shapes
:@А*
dtype0
w
conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_1/bias/m
p
#conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpconv2d_1/bias/m*
_output_shapes	
:А*
dtype0
ѓ
%separable_conv2d_2/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_2/depthwise_kernel/m
®
9separable_conv2d_2/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_2/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
∞
%separable_conv2d_2/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_2/pointwise_kernel/m
©
9separable_conv2d_2/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_2/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_2/bias/m
Д
-separable_conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_2/bias/m*
_output_shapes	
:А*
dtype0
ѓ
%separable_conv2d_3/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_3/depthwise_kernel/m
®
9separable_conv2d_3/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_3/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
∞
%separable_conv2d_3/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_3/pointwise_kernel/m
©
9separable_conv2d_3/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_3/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_3/bias/m
Д
-separable_conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_3/bias/m*
_output_shapes	
:А*
dtype0
ѓ
%separable_conv2d_4/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_4/depthwise_kernel/m
®
9separable_conv2d_4/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_4/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
∞
%separable_conv2d_4/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_4/pointwise_kernel/m
©
9separable_conv2d_4/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_4/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_4/bias/m
Д
-separable_conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_4/bias/m*
_output_shapes	
:А*
dtype0
ѓ
%separable_conv2d_5/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_5/depthwise_kernel/m
®
9separable_conv2d_5/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_5/depthwise_kernel/m*'
_output_shapes
:А*
dtype0
∞
%separable_conv2d_5/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_5/pointwise_kernel/m
©
9separable_conv2d_5/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp%separable_conv2d_5/pointwise_kernel/m*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_5/bias/m
Д
-separable_conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpseparable_conv2d_5/bias/m*
_output_shapes	
:А*
dtype0
И
conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*"
shared_nameconv2d_2/kernel/m
Б
%conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_2/kernel/m*(
_output_shapes
:АА*
dtype0
w
conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_2/bias/m
p
#conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpconv2d_2/bias/m*
_output_shapes	
:А*
dtype0
С
regression_head_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*+
shared_nameregression_head_1/kernel/m
К
.regression_head_1/kernel/m/Read/ReadVariableOpReadVariableOpregression_head_1/kernel/m*
_output_shapes
:	А*
dtype0
И
regression_head_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameregression_head_1/bias/m
Б
,regression_head_1/bias/m/Read/ReadVariableOpReadVariableOpregression_head_1/bias/m*
_output_shapes
:*
dtype0
В
conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d/kernel/v
{
#conv2d/kernel/v/Read/ReadVariableOpReadVariableOpconv2d/kernel/v*&
_output_shapes
:@*
dtype0
r
conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias/v
k
!conv2d/bias/v/Read/ReadVariableOpReadVariableOpconv2d/bias/v*
_output_shapes
:@*
dtype0
™
#separable_conv2d/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#separable_conv2d/depthwise_kernel/v
£
7separable_conv2d/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp#separable_conv2d/depthwise_kernel/v*&
_output_shapes
:@*
dtype0
Ђ
#separable_conv2d/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*4
shared_name%#separable_conv2d/pointwise_kernel/v
§
7separable_conv2d/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp#separable_conv2d/pointwise_kernel/v*'
_output_shapes
:@А*
dtype0
З
separable_conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameseparable_conv2d/bias/v
А
+separable_conv2d/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d/bias/v*
_output_shapes	
:А*
dtype0
ѓ
%separable_conv2d_1/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_1/depthwise_kernel/v
®
9separable_conv2d_1/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_1/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
∞
%separable_conv2d_1/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_1/pointwise_kernel/v
©
9separable_conv2d_1/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_1/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_1/bias/v
Д
-separable_conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_1/bias/v*
_output_shapes	
:А*
dtype0
З
conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*"
shared_nameconv2d_1/kernel/v
А
%conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_1/kernel/v*'
_output_shapes
:@А*
dtype0
w
conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_1/bias/v
p
#conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpconv2d_1/bias/v*
_output_shapes	
:А*
dtype0
ѓ
%separable_conv2d_2/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_2/depthwise_kernel/v
®
9separable_conv2d_2/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_2/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
∞
%separable_conv2d_2/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_2/pointwise_kernel/v
©
9separable_conv2d_2/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_2/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_2/bias/v
Д
-separable_conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_2/bias/v*
_output_shapes	
:А*
dtype0
ѓ
%separable_conv2d_3/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_3/depthwise_kernel/v
®
9separable_conv2d_3/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_3/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
∞
%separable_conv2d_3/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_3/pointwise_kernel/v
©
9separable_conv2d_3/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_3/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_3/bias/v
Д
-separable_conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_3/bias/v*
_output_shapes	
:А*
dtype0
ѓ
%separable_conv2d_4/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_4/depthwise_kernel/v
®
9separable_conv2d_4/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_4/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
∞
%separable_conv2d_4/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_4/pointwise_kernel/v
©
9separable_conv2d_4/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_4/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_4/bias/v
Д
-separable_conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_4/bias/v*
_output_shapes	
:А*
dtype0
ѓ
%separable_conv2d_5/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%separable_conv2d_5/depthwise_kernel/v
®
9separable_conv2d_5/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_5/depthwise_kernel/v*'
_output_shapes
:А*
dtype0
∞
%separable_conv2d_5/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*6
shared_name'%separable_conv2d_5/pointwise_kernel/v
©
9separable_conv2d_5/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp%separable_conv2d_5/pointwise_kernel/v*(
_output_shapes
:АА*
dtype0
Л
separable_conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameseparable_conv2d_5/bias/v
Д
-separable_conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpseparable_conv2d_5/bias/v*
_output_shapes	
:А*
dtype0
И
conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*"
shared_nameconv2d_2/kernel/v
Б
%conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_2/kernel/v*(
_output_shapes
:АА*
dtype0
w
conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_2/bias/v
p
#conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpconv2d_2/bias/v*
_output_shapes	
:А*
dtype0
С
regression_head_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*+
shared_nameregression_head_1/kernel/v
К
.regression_head_1/kernel/v/Read/ReadVariableOpReadVariableOpregression_head_1/kernel/v*
_output_shapes
:	А*
dtype0
И
regression_head_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameregression_head_1/bias/v
Б
,regression_head_1/bias/v/Read/ReadVariableOpReadVariableOpregression_head_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ъП
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*іП
value©ПB•П BЭП
ф
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer-12
layer_with_weights-9
layer-13
layer-14
layer-15
layer_with_weights-10
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
†
state_variables
_broadcast_shape
mean
variance
	count
	variables
trainable_variables
regularization_losses
 	keras_api
h

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
И
'depthwise_kernel
(pointwise_kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
И
.depthwise_kernel
/pointwise_kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
h

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
R
;	variables
<trainable_variables
=regularization_losses
>	keras_api
И
?depthwise_kernel
@pointwise_kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
И
Fdepthwise_kernel
Gpointwise_kernel
Hbias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
R
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
И
Qdepthwise_kernel
Rpointwise_kernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
И
Xdepthwise_kernel
Ypointwise_kernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
R
_	variables
`trainable_variables
aregularization_losses
b	keras_api
h

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
R
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
R
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
h

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
И!m”"m‘'m’(m÷)m„.mЎ/mў0mЏ5mџ6m№?mЁ@mёAmяFmаGmбHmвQmгRmдSmеXmжYmзZmиcmйdmкqmлrmм!vн"vо'vп(vр)vс.vт/vу0vф5vх6vц?vч@vшAvщFvъGvыHvьQvэRvюSv€XvАYvБZvВcvГdvДqvЕrvЖ
ё
0
1
2
!3
"4
'5
(6
)7
.8
/9
010
511
612
?13
@14
A15
F16
G17
H18
Q19
R20
S21
X22
Y23
Z24
c25
d26
q27
r28
∆
!0
"1
'2
(3
)4
.5
/6
07
58
69
?10
@11
A12
F13
G14
H15
Q16
R17
S18
X19
Y20
Z21
c22
d23
q24
r25
 
Ъ
wmetrics
	variables
trainable_variables
xnon_trainable_variables
ylayer_regularization_losses
regularization_losses

zlayers
 
#
mean
variance
	count
 
\Z
VARIABLE_VALUEnormalization/mean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEnormalization/variance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEnormalization/count5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
Ъ
{metrics
	variables
trainable_variables
|non_trainable_variables
}layer_regularization_losses
regularization_losses

~layers
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
Э
metrics
#	variables
$trainable_variables
Аnon_trainable_variables
 Бlayer_regularization_losses
%regularization_losses
Вlayers
wu
VARIABLE_VALUE!separable_conv2d/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE!separable_conv2d/pointwise_kernel@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEseparable_conv2d/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
)2

'0
(1
)2
 
Ю
Гmetrics
*	variables
+trainable_variables
Дnon_trainable_variables
 Еlayer_regularization_losses
,regularization_losses
Жlayers
yw
VARIABLE_VALUE#separable_conv2d_1/depthwise_kernel@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_1/pointwise_kernel@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
02

.0
/1
02
 
Ю
Зmetrics
1	variables
2trainable_variables
Иnon_trainable_variables
 Йlayer_regularization_losses
3regularization_losses
Кlayers
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61

50
61
 
Ю
Лmetrics
7	variables
8trainable_variables
Мnon_trainable_variables
 Нlayer_regularization_losses
9regularization_losses
Оlayers
 
 
 
Ю
Пmetrics
;	variables
<trainable_variables
Рnon_trainable_variables
 Сlayer_regularization_losses
=regularization_losses
Тlayers
yw
VARIABLE_VALUE#separable_conv2d_2/depthwise_kernel@layer_with_weights-5/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_2/pointwise_kernel@layer_with_weights-5/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
A2

?0
@1
A2
 
Ю
Уmetrics
B	variables
Ctrainable_variables
Фnon_trainable_variables
 Хlayer_regularization_losses
Dregularization_losses
Цlayers
yw
VARIABLE_VALUE#separable_conv2d_3/depthwise_kernel@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_3/pointwise_kernel@layer_with_weights-6/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1
H2

F0
G1
H2
 
Ю
Чmetrics
I	variables
Jtrainable_variables
Шnon_trainable_variables
 Щlayer_regularization_losses
Kregularization_losses
Ъlayers
 
 
 
Ю
Ыmetrics
M	variables
Ntrainable_variables
Ьnon_trainable_variables
 Эlayer_regularization_losses
Oregularization_losses
Юlayers
yw
VARIABLE_VALUE#separable_conv2d_4/depthwise_kernel@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_4/pointwise_kernel@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
S2

Q0
R1
S2
 
Ю
Яmetrics
T	variables
Utrainable_variables
†non_trainable_variables
 °layer_regularization_losses
Vregularization_losses
Ґlayers
yw
VARIABLE_VALUE#separable_conv2d_5/depthwise_kernel@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_5/pointwise_kernel@layer_with_weights-8/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1
Z2

X0
Y1
Z2
 
Ю
£metrics
[	variables
\trainable_variables
§non_trainable_variables
 •layer_regularization_losses
]regularization_losses
¶layers
 
 
 
Ю
Іmetrics
_	variables
`trainable_variables
®non_trainable_variables
 ©layer_regularization_losses
aregularization_losses
™layers
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

c0
d1

c0
d1
 
Ю
Ђmetrics
e	variables
ftrainable_variables
ђnon_trainable_variables
 ≠layer_regularization_losses
gregularization_losses
Ѓlayers
 
 
 
Ю
ѓmetrics
i	variables
jtrainable_variables
∞non_trainable_variables
 ±layer_regularization_losses
kregularization_losses
≤layers
 
 
 
Ю
≥metrics
m	variables
ntrainable_variables
іnon_trainable_variables
 µlayer_regularization_losses
oregularization_losses
ґlayers
ec
VARIABLE_VALUEregression_head_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEregression_head_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1

q0
r1
 
Ю
Јmetrics
s	variables
ttrainable_variables
Єnon_trainable_variables
 єlayer_regularization_losses
uregularization_losses
Їlayers

ї0
Љ1

0
1
2
 
~
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
 

0
1
2
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


љtotal

Њcount
њ
_fn_kwargs
ј	variables
Ѕtrainable_variables
¬regularization_losses
√	keras_api


ƒtotal

≈count
∆
_fn_kwargs
«	variables
»trainable_variables
…regularization_losses
 	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

љ0
Њ1
 
 
°
Ћmetrics
ј	variables
Ѕtrainable_variables
ћnon_trainable_variables
 Ќlayer_regularization_losses
¬regularization_losses
ќlayers
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ƒ0
≈1
 
 
°
ѕmetrics
«	variables
»trainable_variables
–non_trainable_variables
 —layer_regularization_losses
…regularization_losses
“layers
 

љ0
Њ1
 
 
 

ƒ0
≈1
 
 
wu
VARIABLE_VALUEconv2d/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEconv2d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE#separable_conv2d/depthwise_kernel/m\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE#separable_conv2d/pointwise_kernel/m\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEseparable_conv2d/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_1/depthwise_kernel/m\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_1/pointwise_kernel/m\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_2/depthwise_kernel/m\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_2/pointwise_kernel/m\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_3/depthwise_kernel/m\layer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_3/pointwise_kernel/m\layer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_4/depthwise_kernel/m\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_4/pointwise_kernel/m\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_4/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_5/depthwise_kernel/m\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_5/pointwise_kernel/m\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_5/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_2/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_2/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEregression_head_1/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEregression_head_1/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEconv2d/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEconv2d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE#separable_conv2d/depthwise_kernel/v\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE#separable_conv2d/pointwise_kernel/v\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEseparable_conv2d/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_1/depthwise_kernel/v\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_1/pointwise_kernel/v\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_2/depthwise_kernel/v\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_2/pointwise_kernel/v\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_3/depthwise_kernel/v\layer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_3/pointwise_kernel/v\layer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_4/depthwise_kernel/v\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_4/pointwise_kernel/v\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_4/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_5/depthwise_kernel/v\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE%separable_conv2d_5/pointwise_kernel/v\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEseparable_conv2d_5/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_2/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_2/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEregression_head_1/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEregression_head_1/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
О
serving_default_input_1Placeholder*1
_output_shapes
:€€€€€€€€€АА*
dtype0*&
shape:€€€€€€€€€АА
ƒ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1normalization/meannormalization/varianceconv2d/kernelconv2d/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasconv2d_1/kernelconv2d_1/bias#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/bias#separable_conv2d_5/depthwise_kernel#separable_conv2d_5/pointwise_kernelseparable_conv2d_5/biasconv2d_2/kernelconv2d_2/biasregression_head_1/kernelregression_head_1/bias*(
Tin!
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*.
f)R'
%__inference_signature_wrapper_9446004
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ё"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&normalization/mean/Read/ReadVariableOp*normalization/variance/Read/ReadVariableOp'normalization/count/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp5separable_conv2d/depthwise_kernel/Read/ReadVariableOp5separable_conv2d/pointwise_kernel/Read/ReadVariableOp)separable_conv2d/bias/Read/ReadVariableOp7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_1/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp7separable_conv2d_2/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_2/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_2/bias/Read/ReadVariableOp7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_3/bias/Read/ReadVariableOp7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_4/bias/Read/ReadVariableOp7separable_conv2d_5/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_5/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_5/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp,regression_head_1/kernel/Read/ReadVariableOp*regression_head_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp#conv2d/kernel/m/Read/ReadVariableOp!conv2d/bias/m/Read/ReadVariableOp7separable_conv2d/depthwise_kernel/m/Read/ReadVariableOp7separable_conv2d/pointwise_kernel/m/Read/ReadVariableOp+separable_conv2d/bias/m/Read/ReadVariableOp9separable_conv2d_1/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_1/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_1/bias/m/Read/ReadVariableOp%conv2d_1/kernel/m/Read/ReadVariableOp#conv2d_1/bias/m/Read/ReadVariableOp9separable_conv2d_2/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_2/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_2/bias/m/Read/ReadVariableOp9separable_conv2d_3/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_3/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_3/bias/m/Read/ReadVariableOp9separable_conv2d_4/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_4/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_4/bias/m/Read/ReadVariableOp9separable_conv2d_5/depthwise_kernel/m/Read/ReadVariableOp9separable_conv2d_5/pointwise_kernel/m/Read/ReadVariableOp-separable_conv2d_5/bias/m/Read/ReadVariableOp%conv2d_2/kernel/m/Read/ReadVariableOp#conv2d_2/bias/m/Read/ReadVariableOp.regression_head_1/kernel/m/Read/ReadVariableOp,regression_head_1/bias/m/Read/ReadVariableOp#conv2d/kernel/v/Read/ReadVariableOp!conv2d/bias/v/Read/ReadVariableOp7separable_conv2d/depthwise_kernel/v/Read/ReadVariableOp7separable_conv2d/pointwise_kernel/v/Read/ReadVariableOp+separable_conv2d/bias/v/Read/ReadVariableOp9separable_conv2d_1/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_1/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_1/bias/v/Read/ReadVariableOp%conv2d_1/kernel/v/Read/ReadVariableOp#conv2d_1/bias/v/Read/ReadVariableOp9separable_conv2d_2/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_2/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_2/bias/v/Read/ReadVariableOp9separable_conv2d_3/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_3/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_3/bias/v/Read/ReadVariableOp9separable_conv2d_4/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_4/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_4/bias/v/Read/ReadVariableOp9separable_conv2d_5/depthwise_kernel/v/Read/ReadVariableOp9separable_conv2d_5/pointwise_kernel/v/Read/ReadVariableOp-separable_conv2d_5/bias/v/Read/ReadVariableOp%conv2d_2/kernel/v/Read/ReadVariableOp#conv2d_2/bias/v/Read/ReadVariableOp.regression_head_1/kernel/v/Read/ReadVariableOp,regression_head_1/bias/v/Read/ReadVariableOpConst*b
Tin[
Y2W*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *1
config_proto!

CPU

GPU2	 *0,1J 8*)
f$R"
 __inference__traced_save_9446660
і
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenormalization/meannormalization/variancenormalization/countconv2d/kernelconv2d/bias!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/biasconv2d_1/kernelconv2d_1/bias#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/bias#separable_conv2d_5/depthwise_kernel#separable_conv2d_5/pointwise_kernelseparable_conv2d_5/biasconv2d_2/kernelconv2d_2/biasregression_head_1/kernelregression_head_1/biastotalcounttotal_1count_1conv2d/kernel/mconv2d/bias/m#separable_conv2d/depthwise_kernel/m#separable_conv2d/pointwise_kernel/mseparable_conv2d/bias/m%separable_conv2d_1/depthwise_kernel/m%separable_conv2d_1/pointwise_kernel/mseparable_conv2d_1/bias/mconv2d_1/kernel/mconv2d_1/bias/m%separable_conv2d_2/depthwise_kernel/m%separable_conv2d_2/pointwise_kernel/mseparable_conv2d_2/bias/m%separable_conv2d_3/depthwise_kernel/m%separable_conv2d_3/pointwise_kernel/mseparable_conv2d_3/bias/m%separable_conv2d_4/depthwise_kernel/m%separable_conv2d_4/pointwise_kernel/mseparable_conv2d_4/bias/m%separable_conv2d_5/depthwise_kernel/m%separable_conv2d_5/pointwise_kernel/mseparable_conv2d_5/bias/mconv2d_2/kernel/mconv2d_2/bias/mregression_head_1/kernel/mregression_head_1/bias/mconv2d/kernel/vconv2d/bias/v#separable_conv2d/depthwise_kernel/v#separable_conv2d/pointwise_kernel/vseparable_conv2d/bias/v%separable_conv2d_1/depthwise_kernel/v%separable_conv2d_1/pointwise_kernel/vseparable_conv2d_1/bias/vconv2d_1/kernel/vconv2d_1/bias/v%separable_conv2d_2/depthwise_kernel/v%separable_conv2d_2/pointwise_kernel/vseparable_conv2d_2/bias/v%separable_conv2d_3/depthwise_kernel/v%separable_conv2d_3/pointwise_kernel/vseparable_conv2d_3/bias/v%separable_conv2d_4/depthwise_kernel/v%separable_conv2d_4/pointwise_kernel/vseparable_conv2d_4/bias/v%separable_conv2d_5/depthwise_kernel/v%separable_conv2d_5/pointwise_kernel/vseparable_conv2d_5/bias/vconv2d_2/kernel/vconv2d_2/bias/vregression_head_1/kernel/vregression_head_1/bias/v*a
TinZ
X2V*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *1
config_proto!

CPU

GPU2	 *0,1J 8*,
f'R%
#__inference__traced_restore_9446927ЛХ
©
ў
4__inference_separable_conv2d_5_layer_call_fn_9445580

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_94455712
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
“
Q
%__inference_add_layer_call_fn_9446340
inputs_0
inputs_1
identity≈
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_94456762
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:€€€€€€€€€@@А:€€€€€€€€€@@А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
ъ
n
B__inference_add_1_layer_call_and_return_conditional_losses_9446346
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:€€€€€€€€€@@А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:€€€€€€€€€@@А:€€€€€€€€€@@А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
љ
–
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_9445447

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐseparable_conv2d/ReadVariableOpҐ!separable_conv2d/ReadVariableOp_1і
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOpї
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateч
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
2
separable_conv2d/depthwiseф
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOp•
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
Seluа
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
ъ
n
B__inference_add_2_layer_call_and_return_conditional_losses_9446358
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:€€€€€€€€€  А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:€€€€€€€€€  А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:€€€€€€€€€  А:€€€€€€€€€  А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
Х
д
'__inference_model_layer_call_fn_9445970
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28
identityИҐStatefulPartitionedCallщ	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28*(
Tin!
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_94459392
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ґ
_input_shapesР
Н:€€€€€€€€€АА::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
р
j
@__inference_add_layer_call_and_return_conditional_losses_9445676

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:€€€€€€€€€@@А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:€€€€€€€€€@@А:€€€€€€€€€@@А:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
т
l
B__inference_add_2_layer_call_and_return_conditional_losses_9445726

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:€€€€€€€€€  А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:€€€€€€€€€  А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:€€€€€€€€€  А:€€€€€€€€€  А:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
у
в
%__inference_signature_wrapper_9446004
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28
identityИҐStatefulPartitionedCallў	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28*(
Tin!
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*+
f&R$
"__inference__wrapped_model_94453832
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ґ
_input_shapesР
Н:€€€€€€€€€АА::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
џ
й
J__inference_normalization_layer_call_and_return_conditional_losses_9445644

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identityИҐReshape/ReadVariableOpҐReshape_1/ReadVariableOpМ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shapeЖ
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
ReshapeТ
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shapeО
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1g
subSubinputsReshape:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
subY
SqrtSqrtReshape_1:output:0*
T0*&
_output_shapes
:2
Sqrtl
truedivRealDivsub:z:0Sqrt:y:0*
T0*1
_output_shapes
:€€€€€€€€€АА2	
truedivЭ
IdentityIdentitytruediv:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:& "
 
_user_specified_nameinputs
ћ
Ђ
*__inference_conv2d_2_layer_call_fn_9445612

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*1
config_proto!

CPU

GPU2	 *0,1J 8*N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_94456042
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
п
q
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_9445619

inputs
identityБ
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
О
і
3__inference_regression_head_1_layer_call_fn_9446381

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*W
fRRP
N__inference_regression_head_1_layer_call_and_return_conditional_losses_94457462
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ћ√
Щ
B__inference_model_layer_call_and_return_conditional_losses_9446122

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource=
9separable_conv2d_separable_conv2d_readvariableop_resource?
;separable_conv2d_separable_conv2d_readvariableop_1_resource4
0separable_conv2d_biasadd_readvariableop_resource?
;separable_conv2d_1_separable_conv2d_readvariableop_resourceA
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_1_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource?
;separable_conv2d_2_separable_conv2d_readvariableop_resourceA
=separable_conv2d_2_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_2_biasadd_readvariableop_resource?
;separable_conv2d_3_separable_conv2d_readvariableop_resourceA
=separable_conv2d_3_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_3_biasadd_readvariableop_resource?
;separable_conv2d_4_separable_conv2d_readvariableop_resourceA
=separable_conv2d_4_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_4_biasadd_readvariableop_resource?
;separable_conv2d_5_separable_conv2d_readvariableop_resourceA
=separable_conv2d_5_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_5_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource4
0regression_head_1_matmul_readvariableop_resource5
1regression_head_1_biasadd_readvariableop_resource
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐ$normalization/Reshape/ReadVariableOpҐ&normalization/Reshape_1/ReadVariableOpҐ(regression_head_1/BiasAdd/ReadVariableOpҐ'regression_head_1/MatMul/ReadVariableOpҐ'separable_conv2d/BiasAdd/ReadVariableOpҐ0separable_conv2d/separable_conv2d/ReadVariableOpҐ2separable_conv2d/separable_conv2d/ReadVariableOp_1Ґ)separable_conv2d_1/BiasAdd/ReadVariableOpҐ2separable_conv2d_1/separable_conv2d/ReadVariableOpҐ4separable_conv2d_1/separable_conv2d/ReadVariableOp_1Ґ)separable_conv2d_2/BiasAdd/ReadVariableOpҐ2separable_conv2d_2/separable_conv2d/ReadVariableOpҐ4separable_conv2d_2/separable_conv2d/ReadVariableOp_1Ґ)separable_conv2d_3/BiasAdd/ReadVariableOpҐ2separable_conv2d_3/separable_conv2d/ReadVariableOpҐ4separable_conv2d_3/separable_conv2d/ReadVariableOp_1Ґ)separable_conv2d_4/BiasAdd/ReadVariableOpҐ2separable_conv2d_4/separable_conv2d/ReadVariableOpҐ4separable_conv2d_4/separable_conv2d/ReadVariableOp_1Ґ)separable_conv2d_5/BiasAdd/ReadVariableOpҐ2separable_conv2d_5/separable_conv2d/ReadVariableOpҐ4separable_conv2d_5/separable_conv2d/ReadVariableOp_1ґ
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOpУ
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shapeЊ
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/ReshapeЉ
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOpЧ
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shape∆
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1С
normalization/subSubinputsnormalization/Reshape:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
normalization/subГ
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrt§
normalization/truedivRealDivnormalization/sub:z:0normalization/Sqrt:y:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
normalization/truediv™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOpЋ
conv2d/Conv2DConv2Dnormalization/truediv:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@*
paddingSAME*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2
conv2d/BiasAddu
conv2d/SeluSeluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2
conv2d/Seluж
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype022
0separable_conv2d/separable_conv2d/ReadVariableOpн
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype024
2separable_conv2d/separable_conv2d/ReadVariableOp_1Ђ
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2)
'separable_conv2d/separable_conv2d/Shape≥
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      21
/separable_conv2d/separable_conv2d/dilation_rate™
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeconv2d/Selu:activations:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@*
paddingSAME*
strides
2-
+separable_conv2d/separable_conv2d/depthwise¶
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2#
!separable_conv2d/separable_conv2dј
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02)
'separable_conv2d/BiasAdd/ReadVariableOp„
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d/BiasAddФ
separable_conv2d/SeluSelu!separable_conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d/Seluн
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_1/separable_conv2d/ReadVariableOpф
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ѓ
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2+
)separable_conv2d_1/separable_conv2d/ShapeЈ
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_1/separable_conv2d/dilation_rateї
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Selu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2/
-separable_conv2d_1/separable_conv2d/depthwiseЃ
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2%
#separable_conv2d_1/separable_conv2d∆
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_1/BiasAdd/ReadVariableOpя
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_1/BiasAddЪ
separable_conv2d_1/SeluSelu#separable_conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_1/Selu±
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_1/Conv2D/ReadVariableOp“
conv2d_1/Conv2DConv2Dconv2d/Selu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2
conv2d_1/Conv2D®
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp≠
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
conv2d_1/BiasAddШ
add/addAddV2%separable_conv2d_1/Selu:activations:0conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2	
add/addн
2separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_2_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_2/separable_conv2d/ReadVariableOpф
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_2_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ѓ
)separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2+
)separable_conv2d_2/separable_conv2d/ShapeЈ
1separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_2/separable_conv2d/dilation_rate£
-separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativeadd/add:z:0:separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2/
-separable_conv2d_2/separable_conv2d/depthwiseЃ
#separable_conv2d_2/separable_conv2dConv2D6separable_conv2d_2/separable_conv2d/depthwise:output:0<separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2%
#separable_conv2d_2/separable_conv2d∆
)separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_2/BiasAdd/ReadVariableOpя
separable_conv2d_2/BiasAddBiasAdd,separable_conv2d_2/separable_conv2d:output:01separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_2/BiasAddЪ
separable_conv2d_2/SeluSelu#separable_conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_2/Seluн
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_3/separable_conv2d/ReadVariableOpф
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ѓ
)separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2+
)separable_conv2d_3/separable_conv2d/ShapeЈ
1separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_3/separable_conv2d/dilation_rateљ
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_2/Selu:activations:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2/
-separable_conv2d_3/separable_conv2d/depthwiseЃ
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2%
#separable_conv2d_3/separable_conv2d∆
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_3/BiasAdd/ReadVariableOpя
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_3/BiasAddЪ
separable_conv2d_3/SeluSelu#separable_conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_3/SeluО
	add_1/addAddV2%separable_conv2d_3/Selu:activations:0add/add:z:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
	add_1/addн
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_4/separable_conv2d/ReadVariableOpф
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ѓ
)separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2+
)separable_conv2d_4/separable_conv2d/ShapeЈ
1separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_4/separable_conv2d/dilation_rate•
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_1/add:z:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2/
-separable_conv2d_4/separable_conv2d/depthwiseЃ
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2%
#separable_conv2d_4/separable_conv2d∆
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_4/BiasAdd/ReadVariableOpя
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_4/BiasAddЪ
separable_conv2d_4/SeluSelu#separable_conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_4/Seluн
2separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_5_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_5/separable_conv2d/ReadVariableOpф
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_5_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1ѓ
)separable_conv2d_5/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_5/separable_conv2d/ShapeЈ
1separable_conv2d_5/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_5/separable_conv2d/dilation_rateљ
-separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_4/Selu:activations:0:separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2/
-separable_conv2d_5/separable_conv2d/depthwiseЃ
#separable_conv2d_5/separable_conv2dConv2D6separable_conv2d_5/separable_conv2d/depthwise:output:0<separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2%
#separable_conv2d_5/separable_conv2d∆
)separable_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_5/BiasAdd/ReadVariableOpя
separable_conv2d_5/BiasAddBiasAdd,separable_conv2d_5/separable_conv2d:output:01separable_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_5/BiasAddЪ
separable_conv2d_5/SeluSelu#separable_conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_5/SeluЌ
max_pooling2d/MaxPoolMaxPool%separable_conv2d_5/Selu:activations:0*0
_output_shapes
:€€€€€€€€€  А*
ksize
*
paddingSAME*
strides
2
max_pooling2d/MaxPool≤
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02 
conv2d_2/Conv2D/ReadVariableOp∆
conv2d_2/Conv2DConv2Dadd_1/add:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€  А*
paddingSAME*
strides
2
conv2d_2/Conv2D®
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp≠
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€  А2
conv2d_2/BiasAddХ
	add_2/addAddV2max_pooling2d/MaxPool:output:0conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€  А2
	add_2/add≥
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices¬
global_average_pooling2d/MeanMeanadd_2/add:z:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
global_average_pooling2d/Meanƒ
'regression_head_1/MatMul/ReadVariableOpReadVariableOp0regression_head_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02)
'regression_head_1/MatMul/ReadVariableOp…
regression_head_1/MatMulMatMul&global_average_pooling2d/Mean:output:0/regression_head_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
regression_head_1/MatMul¬
(regression_head_1/BiasAdd/ReadVariableOpReadVariableOp1regression_head_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(regression_head_1/BiasAdd/ReadVariableOp…
regression_head_1/BiasAddBiasAdd"regression_head_1/MatMul:product:00regression_head_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
regression_head_1/BiasAddк

IdentityIdentity"regression_head_1/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp)^regression_head_1/BiasAdd/ReadVariableOp(^regression_head_1/MatMul/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*^separable_conv2d_2/BiasAdd/ReadVariableOp3^separable_conv2d_2/separable_conv2d/ReadVariableOp5^separable_conv2d_2/separable_conv2d/ReadVariableOp_1*^separable_conv2d_3/BiasAdd/ReadVariableOp3^separable_conv2d_3/separable_conv2d/ReadVariableOp5^separable_conv2d_3/separable_conv2d/ReadVariableOp_1*^separable_conv2d_4/BiasAdd/ReadVariableOp3^separable_conv2d_4/separable_conv2d/ReadVariableOp5^separable_conv2d_4/separable_conv2d/ReadVariableOp_1*^separable_conv2d_5/BiasAdd/ReadVariableOp3^separable_conv2d_5/separable_conv2d/ReadVariableOp5^separable_conv2d_5/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ґ
_input_shapesР
Н:€€€€€€€€€АА::::::::::::::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2T
(regression_head_1/BiasAdd/ReadVariableOp(regression_head_1/BiasAdd/ReadVariableOp2R
'regression_head_1/MatMul/ReadVariableOp'regression_head_1/MatMul/ReadVariableOp2R
'separable_conv2d/BiasAdd/ReadVariableOp'separable_conv2d/BiasAdd/ReadVariableOp2d
0separable_conv2d/separable_conv2d/ReadVariableOp0separable_conv2d/separable_conv2d/ReadVariableOp2h
2separable_conv2d/separable_conv2d/ReadVariableOp_12separable_conv2d/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_1/BiasAdd/ReadVariableOp)separable_conv2d_1/BiasAdd/ReadVariableOp2h
2separable_conv2d_1/separable_conv2d/ReadVariableOp2separable_conv2d_1/separable_conv2d/ReadVariableOp2l
4separable_conv2d_1/separable_conv2d/ReadVariableOp_14separable_conv2d_1/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_2/BiasAdd/ReadVariableOp)separable_conv2d_2/BiasAdd/ReadVariableOp2h
2separable_conv2d_2/separable_conv2d/ReadVariableOp2separable_conv2d_2/separable_conv2d/ReadVariableOp2l
4separable_conv2d_2/separable_conv2d/ReadVariableOp_14separable_conv2d_2/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_3/BiasAdd/ReadVariableOp)separable_conv2d_3/BiasAdd/ReadVariableOp2h
2separable_conv2d_3/separable_conv2d/ReadVariableOp2separable_conv2d_3/separable_conv2d/ReadVariableOp2l
4separable_conv2d_3/separable_conv2d/ReadVariableOp_14separable_conv2d_3/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_4/BiasAdd/ReadVariableOp)separable_conv2d_4/BiasAdd/ReadVariableOp2h
2separable_conv2d_4/separable_conv2d/ReadVariableOp2separable_conv2d_4/separable_conv2d/ReadVariableOp2l
4separable_conv2d_4/separable_conv2d/ReadVariableOp_14separable_conv2d_4/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_5/BiasAdd/ReadVariableOp)separable_conv2d_5/BiasAdd/ReadVariableOp2h
2separable_conv2d_5/separable_conv2d/ReadVariableOp2separable_conv2d_5/separable_conv2d/ReadVariableOp2l
4separable_conv2d_5/separable_conv2d/ReadVariableOp_14separable_conv2d_5/separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
Х
д
'__inference_model_layer_call_fn_9445889
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28
identityИҐStatefulPartitionedCallщ	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28*(
Tin!
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_94458582
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ґ
_input_shapesР
Н:€€€€€€€€€АА::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
©
ў
4__inference_separable_conv2d_1_layer_call_fn_9445456

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_94454472
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
©
ў
4__inference_separable_conv2d_2_layer_call_fn_9445502

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_94454932
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
©
ў
4__inference_separable_conv2d_4_layer_call_fn_9445554

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_94455452
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ґ[
Џ
B__inference_model_layer_call_and_return_conditional_losses_9445858

inputs0
,normalization_statefulpartitionedcall_args_10
,normalization_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_13
/separable_conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_35
1separable_conv2d_1_statefulpartitionedcall_args_15
1separable_conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_1_statefulpartitionedcall_args_3+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_15
1separable_conv2d_2_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_35
1separable_conv2d_3_statefulpartitionedcall_args_15
1separable_conv2d_3_statefulpartitionedcall_args_25
1separable_conv2d_3_statefulpartitionedcall_args_35
1separable_conv2d_4_statefulpartitionedcall_args_15
1separable_conv2d_4_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_35
1separable_conv2d_5_statefulpartitionedcall_args_15
1separable_conv2d_5_statefulpartitionedcall_args_25
1separable_conv2d_5_statefulpartitionedcall_args_3+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_24
0regression_head_1_statefulpartitionedcall_args_14
0regression_head_1_statefulpartitionedcall_args_2
identityИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ%normalization/StatefulPartitionedCallҐ)regression_head_1/StatefulPartitionedCallҐ(separable_conv2d/StatefulPartitionedCallҐ*separable_conv2d_1/StatefulPartitionedCallҐ*separable_conv2d_2/StatefulPartitionedCallҐ*separable_conv2d_3/StatefulPartitionedCallҐ*separable_conv2d_4/StatefulPartitionedCallҐ*separable_conv2d_5/StatefulPartitionedCall’
%normalization/StatefulPartitionedCallStatefulPartitionedCallinputs,normalization_statefulpartitionedcall_args_1,normalization_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:€€€€€€€€€АА*1
config_proto!

CPU

GPU2	 *0,1J 8*S
fNRL
J__inference_normalization_layer_call_and_return_conditional_losses_94456442'
%normalization/StatefulPartitionedCallЎ
conv2d/StatefulPartitionedCallStatefulPartitionedCall.normalization/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€@@@*1
config_proto!

CPU

GPU2	 *0,1J 8*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_94453962 
conv2d/StatefulPartitionedCallґ
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0/separable_conv2d_statefulpartitionedcall_args_1/separable_conv2d_statefulpartitionedcall_args_2/separable_conv2d_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*V
fQRO
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_94454212*
(separable_conv2d/StatefulPartitionedCallћ
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:01separable_conv2d_1_statefulpartitionedcall_args_11separable_conv2d_1_statefulpartitionedcall_args_21separable_conv2d_1_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_94454472,
*separable_conv2d_1/StatefulPartitionedCall№
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_94454682"
 conv2d_1/StatefulPartitionedCallЩ
add/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_94456762
add/PartitionedCallЈ
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:01separable_conv2d_2_statefulpartitionedcall_args_11separable_conv2d_2_statefulpartitionedcall_args_21separable_conv2d_2_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_94454932,
*separable_conv2d_2/StatefulPartitionedCallќ
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:01separable_conv2d_3_statefulpartitionedcall_args_11separable_conv2d_3_statefulpartitionedcall_args_21separable_conv2d_3_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_94455192,
*separable_conv2d_3/StatefulPartitionedCallТ
add_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_94456992
add_1/PartitionedCallє
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:01separable_conv2d_4_statefulpartitionedcall_args_11separable_conv2d_4_statefulpartitionedcall_args_21separable_conv2d_4_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_94455452,
*separable_conv2d_4/StatefulPartitionedCallќ
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:01separable_conv2d_5_statefulpartitionedcall_args_11separable_conv2d_5_statefulpartitionedcall_args_21separable_conv2d_5_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_94455712,
*separable_conv2d_5/StatefulPartitionedCallЛ
max_pooling2d/PartitionedCallPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_94455862
max_pooling2d/PartitionedCall”
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_94456042"
 conv2d_2/StatefulPartitionedCallТ
add_2/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_94457262
add_2/PartitionedCallП
(global_average_pooling2d/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:€€€€€€€€€А*1
config_proto!

CPU

GPU2	 *0,1J 8*^
fYRW
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_94456192*
(global_average_pooling2d/PartitionedCallК
)regression_head_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:00regression_head_1_statefulpartitionedcall_args_10regression_head_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*W
fRRP
N__inference_regression_head_1_layer_call_and_return_conditional_losses_94457462+
)regression_head_1/StatefulPartitionedCallЌ
IdentityIdentity2regression_head_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall&^normalization/StatefulPartitionedCall*^regression_head_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ґ
_input_shapesР
Н:€€€€€€€€€АА::::::::::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2N
%normalization/StatefulPartitionedCall%normalization/StatefulPartitionedCall2V
)regression_head_1/StatefulPartitionedCall)regression_head_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
х
з
N__inference_regression_head_1_layer_call_and_return_conditional_losses_9446374

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
љ
–
O__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_9445519

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐseparable_conv2d/ReadVariableOpҐ!separable_conv2d/ReadVariableOp_1і
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOpї
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateч
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
2
separable_conv2d/depthwiseф
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOp•
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
Seluа
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
х
з
N__inference_regression_head_1_layer_call_and_return_conditional_losses_9445746

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ц÷
 
"__inference__wrapped_model_9445383
input_17
3model_normalization_reshape_readvariableop_resource9
5model_normalization_reshape_1_readvariableop_resource/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resourceC
?model_separable_conv2d_separable_conv2d_readvariableop_resourceE
Amodel_separable_conv2d_separable_conv2d_readvariableop_1_resource:
6model_separable_conv2d_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_1_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_1_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_1_biasadd_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_2_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_2_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_2_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_3_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_3_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_3_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_4_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_4_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_4_biasadd_readvariableop_resourceE
Amodel_separable_conv2d_5_separable_conv2d_readvariableop_resourceG
Cmodel_separable_conv2d_5_separable_conv2d_readvariableop_1_resource<
8model_separable_conv2d_5_biasadd_readvariableop_resource1
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource:
6model_regression_head_1_matmul_readvariableop_resource;
7model_regression_head_1_biasadd_readvariableop_resource
identityИҐ#model/conv2d/BiasAdd/ReadVariableOpҐ"model/conv2d/Conv2D/ReadVariableOpҐ%model/conv2d_1/BiasAdd/ReadVariableOpҐ$model/conv2d_1/Conv2D/ReadVariableOpҐ%model/conv2d_2/BiasAdd/ReadVariableOpҐ$model/conv2d_2/Conv2D/ReadVariableOpҐ*model/normalization/Reshape/ReadVariableOpҐ,model/normalization/Reshape_1/ReadVariableOpҐ.model/regression_head_1/BiasAdd/ReadVariableOpҐ-model/regression_head_1/MatMul/ReadVariableOpҐ-model/separable_conv2d/BiasAdd/ReadVariableOpҐ6model/separable_conv2d/separable_conv2d/ReadVariableOpҐ8model/separable_conv2d/separable_conv2d/ReadVariableOp_1Ґ/model/separable_conv2d_1/BiasAdd/ReadVariableOpҐ8model/separable_conv2d_1/separable_conv2d/ReadVariableOpҐ:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1Ґ/model/separable_conv2d_2/BiasAdd/ReadVariableOpҐ8model/separable_conv2d_2/separable_conv2d/ReadVariableOpҐ:model/separable_conv2d_2/separable_conv2d/ReadVariableOp_1Ґ/model/separable_conv2d_3/BiasAdd/ReadVariableOpҐ8model/separable_conv2d_3/separable_conv2d/ReadVariableOpҐ:model/separable_conv2d_3/separable_conv2d/ReadVariableOp_1Ґ/model/separable_conv2d_4/BiasAdd/ReadVariableOpҐ8model/separable_conv2d_4/separable_conv2d/ReadVariableOpҐ:model/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Ґ/model/separable_conv2d_5/BiasAdd/ReadVariableOpҐ8model/separable_conv2d_5/separable_conv2d/ReadVariableOpҐ:model/separable_conv2d_5/separable_conv2d/ReadVariableOp_1»
*model/normalization/Reshape/ReadVariableOpReadVariableOp3model_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02,
*model/normalization/Reshape/ReadVariableOpЯ
!model/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2#
!model/normalization/Reshape/shape÷
model/normalization/ReshapeReshape2model/normalization/Reshape/ReadVariableOp:value:0*model/normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
model/normalization/Reshapeќ
,model/normalization/Reshape_1/ReadVariableOpReadVariableOp5model_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,model/normalization/Reshape_1/ReadVariableOp£
#model/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#model/normalization/Reshape_1/shapeё
model/normalization/Reshape_1Reshape4model/normalization/Reshape_1/ReadVariableOp:value:0,model/normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
model/normalization/Reshape_1§
model/normalization/subSubinput_1$model/normalization/Reshape:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
model/normalization/subХ
model/normalization/SqrtSqrt&model/normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
model/normalization/SqrtЉ
model/normalization/truedivRealDivmodel/normalization/sub:z:0model/normalization/Sqrt:y:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
model/normalization/truedivЉ
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"model/conv2d/Conv2D/ReadVariableOpг
model/conv2d/Conv2DConv2Dmodel/normalization/truediv:z:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@*
paddingSAME*
strides
2
model/conv2d/Conv2D≥
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOpЉ
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2
model/conv2d/BiasAddЗ
model/conv2d/SeluSelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2
model/conv2d/Seluш
6model/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp?model_separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype028
6model/separable_conv2d/separable_conv2d/ReadVariableOp€
8model/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpAmodel_separable_conv2d_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype02:
8model/separable_conv2d/separable_conv2d/ReadVariableOp_1Ј
-model/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2/
-model/separable_conv2d/separable_conv2d/Shapeњ
5model/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      27
5model/separable_conv2d/separable_conv2d/dilation_rate¬
1model/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativemodel/conv2d/Selu:activations:0>model/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@*
paddingSAME*
strides
23
1model/separable_conv2d/separable_conv2d/depthwiseЊ
'model/separable_conv2d/separable_conv2dConv2D:model/separable_conv2d/separable_conv2d/depthwise:output:0@model/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2)
'model/separable_conv2d/separable_conv2d“
-model/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp6model_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-model/separable_conv2d/BiasAdd/ReadVariableOpп
model/separable_conv2d/BiasAddBiasAdd0model/separable_conv2d/separable_conv2d:output:05model/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2 
model/separable_conv2d/BiasAdd¶
model/separable_conv2d/SeluSelu'model/separable_conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
model/separable_conv2d/Selu€
8model/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_1_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_1/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ї
/model/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      21
/model/separable_conv2d_1/separable_conv2d/Shape√
7model/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_1/separable_conv2d/dilation_rate”
3model/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative)model/separable_conv2d/Selu:activations:0@model/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
25
3model/separable_conv2d_1/separable_conv2d/depthwise∆
)model/separable_conv2d_1/separable_conv2dConv2D<model/separable_conv2d_1/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2+
)model/separable_conv2d_1/separable_conv2dЎ
/model/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_1/BiasAdd/ReadVariableOpч
 model/separable_conv2d_1/BiasAddBiasAdd2model/separable_conv2d_1/separable_conv2d:output:07model/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2"
 model/separable_conv2d_1/BiasAddђ
model/separable_conv2d_1/SeluSelu)model/separable_conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
model/separable_conv2d_1/Selu√
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOpк
model/conv2d_1/Conv2DConv2Dmodel/conv2d/Selu:activations:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2
model/conv2d_1/Conv2DЇ
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp≈
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
model/conv2d_1/BiasAdd∞
model/add/addAddV2+model/separable_conv2d_1/Selu:activations:0model/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
model/add/add€
8model/separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_2_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_2/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_2_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ї
/model/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      21
/model/separable_conv2d_2/separable_conv2d/Shape√
7model/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_2/separable_conv2d/dilation_rateї
3model/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativemodel/add/add:z:0@model/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
25
3model/separable_conv2d_2/separable_conv2d/depthwise∆
)model/separable_conv2d_2/separable_conv2dConv2D<model/separable_conv2d_2/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2+
)model/separable_conv2d_2/separable_conv2dЎ
/model/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_2/BiasAdd/ReadVariableOpч
 model/separable_conv2d_2/BiasAddBiasAdd2model/separable_conv2d_2/separable_conv2d:output:07model/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2"
 model/separable_conv2d_2/BiasAddђ
model/separable_conv2d_2/SeluSelu)model/separable_conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
model/separable_conv2d_2/Selu€
8model/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_3_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_3/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ї
/model/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      21
/model/separable_conv2d_3/separable_conv2d/Shape√
7model/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_3/separable_conv2d/dilation_rate’
3model/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative+model/separable_conv2d_2/Selu:activations:0@model/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
25
3model/separable_conv2d_3/separable_conv2d/depthwise∆
)model/separable_conv2d_3/separable_conv2dConv2D<model/separable_conv2d_3/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2+
)model/separable_conv2d_3/separable_conv2dЎ
/model/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_3/BiasAdd/ReadVariableOpч
 model/separable_conv2d_3/BiasAddBiasAdd2model/separable_conv2d_3/separable_conv2d:output:07model/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2"
 model/separable_conv2d_3/BiasAddђ
model/separable_conv2d_3/SeluSelu)model/separable_conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
model/separable_conv2d_3/Selu¶
model/add_1/addAddV2+model/separable_conv2d_3/Selu:activations:0model/add/add:z:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
model/add_1/add€
8model/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_4_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_4/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_4_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ї
/model/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      21
/model/separable_conv2d_4/separable_conv2d/Shape√
7model/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_4/separable_conv2d/dilation_rateљ
3model/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativemodel/add_1/add:z:0@model/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
25
3model/separable_conv2d_4/separable_conv2d/depthwise∆
)model/separable_conv2d_4/separable_conv2dConv2D<model/separable_conv2d_4/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2+
)model/separable_conv2d_4/separable_conv2dЎ
/model/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_4/BiasAdd/ReadVariableOpч
 model/separable_conv2d_4/BiasAddBiasAdd2model/separable_conv2d_4/separable_conv2d:output:07model/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2"
 model/separable_conv2d_4/BiasAddђ
model/separable_conv2d_4/SeluSelu)model/separable_conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
model/separable_conv2d_4/Selu€
8model/separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOpAmodel_separable_conv2d_5_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02:
8model/separable_conv2d_5/separable_conv2d/ReadVariableOpЖ
:model/separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_separable_conv2d_5_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02<
:model/separable_conv2d_5/separable_conv2d/ReadVariableOp_1ї
/model/separable_conv2d_5/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model/separable_conv2d_5/separable_conv2d/Shape√
7model/separable_conv2d_5/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/separable_conv2d_5/separable_conv2d/dilation_rate’
3model/separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative+model/separable_conv2d_4/Selu:activations:0@model/separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
25
3model/separable_conv2d_5/separable_conv2d/depthwise∆
)model/separable_conv2d_5/separable_conv2dConv2D<model/separable_conv2d_5/separable_conv2d/depthwise:output:0Bmodel/separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2+
)model/separable_conv2d_5/separable_conv2dЎ
/model/separable_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp8model_separable_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/model/separable_conv2d_5/BiasAdd/ReadVariableOpч
 model/separable_conv2d_5/BiasAddBiasAdd2model/separable_conv2d_5/separable_conv2d:output:07model/separable_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2"
 model/separable_conv2d_5/BiasAddђ
model/separable_conv2d_5/SeluSelu)model/separable_conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
model/separable_conv2d_5/Seluя
model/max_pooling2d/MaxPoolMaxPool+model/separable_conv2d_5/Selu:activations:0*0
_output_shapes
:€€€€€€€€€  А*
ksize
*
paddingSAME*
strides
2
model/max_pooling2d/MaxPoolƒ
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOpё
model/conv2d_2/Conv2DConv2Dmodel/add_1/add:z:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€  А*
paddingSAME*
strides
2
model/conv2d_2/Conv2DЇ
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp≈
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€  А2
model/conv2d_2/BiasAdd≠
model/add_2/addAddV2$model/max_pooling2d/MaxPool:output:0model/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€  А2
model/add_2/addњ
5model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5model/global_average_pooling2d/Mean/reduction_indicesЏ
#model/global_average_pooling2d/MeanMeanmodel/add_2/add:z:0>model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2%
#model/global_average_pooling2d/Mean÷
-model/regression_head_1/MatMul/ReadVariableOpReadVariableOp6model_regression_head_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02/
-model/regression_head_1/MatMul/ReadVariableOpб
model/regression_head_1/MatMulMatMul,model/global_average_pooling2d/Mean:output:05model/regression_head_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2 
model/regression_head_1/MatMul‘
.model/regression_head_1/BiasAdd/ReadVariableOpReadVariableOp7model_regression_head_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.model/regression_head_1/BiasAdd/ReadVariableOpб
model/regression_head_1/BiasAddBiasAdd(model/regression_head_1/MatMul:product:06model/regression_head_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2!
model/regression_head_1/BiasAddШ
IdentityIdentity(model/regression_head_1/BiasAdd:output:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp+^model/normalization/Reshape/ReadVariableOp-^model/normalization/Reshape_1/ReadVariableOp/^model/regression_head_1/BiasAdd/ReadVariableOp.^model/regression_head_1/MatMul/ReadVariableOp.^model/separable_conv2d/BiasAdd/ReadVariableOp7^model/separable_conv2d/separable_conv2d/ReadVariableOp9^model/separable_conv2d/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_1/BiasAdd/ReadVariableOp9^model/separable_conv2d_1/separable_conv2d/ReadVariableOp;^model/separable_conv2d_1/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_2/BiasAdd/ReadVariableOp9^model/separable_conv2d_2/separable_conv2d/ReadVariableOp;^model/separable_conv2d_2/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_3/BiasAdd/ReadVariableOp9^model/separable_conv2d_3/separable_conv2d/ReadVariableOp;^model/separable_conv2d_3/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_4/BiasAdd/ReadVariableOp9^model/separable_conv2d_4/separable_conv2d/ReadVariableOp;^model/separable_conv2d_4/separable_conv2d/ReadVariableOp_10^model/separable_conv2d_5/BiasAdd/ReadVariableOp9^model/separable_conv2d_5/separable_conv2d/ReadVariableOp;^model/separable_conv2d_5/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ґ
_input_shapesР
Н:€€€€€€€€€АА::::::::::::::::::::::::::::2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2X
*model/normalization/Reshape/ReadVariableOp*model/normalization/Reshape/ReadVariableOp2\
,model/normalization/Reshape_1/ReadVariableOp,model/normalization/Reshape_1/ReadVariableOp2`
.model/regression_head_1/BiasAdd/ReadVariableOp.model/regression_head_1/BiasAdd/ReadVariableOp2^
-model/regression_head_1/MatMul/ReadVariableOp-model/regression_head_1/MatMul/ReadVariableOp2^
-model/separable_conv2d/BiasAdd/ReadVariableOp-model/separable_conv2d/BiasAdd/ReadVariableOp2p
6model/separable_conv2d/separable_conv2d/ReadVariableOp6model/separable_conv2d/separable_conv2d/ReadVariableOp2t
8model/separable_conv2d/separable_conv2d/ReadVariableOp_18model/separable_conv2d/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_1/BiasAdd/ReadVariableOp/model/separable_conv2d_1/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_1/separable_conv2d/ReadVariableOp8model/separable_conv2d_1/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_1/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_2/BiasAdd/ReadVariableOp/model/separable_conv2d_2/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_2/separable_conv2d/ReadVariableOp8model/separable_conv2d_2/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_2/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_3/BiasAdd/ReadVariableOp/model/separable_conv2d_3/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_3/separable_conv2d/ReadVariableOp8model/separable_conv2d_3/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_3/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_4/BiasAdd/ReadVariableOp/model/separable_conv2d_4/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_4/separable_conv2d/ReadVariableOp8model/separable_conv2d_4/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_4/separable_conv2d/ReadVariableOp_12b
/model/separable_conv2d_5/BiasAdd/ReadVariableOp/model/separable_conv2d_5/BiasAdd/ReadVariableOp2t
8model/separable_conv2d_5/separable_conv2d/ReadVariableOp8model/separable_conv2d_5/separable_conv2d/ReadVariableOp2x
:model/separable_conv2d_5/separable_conv2d/ReadVariableOp_1:model/separable_conv2d_5/separable_conv2d/ReadVariableOp_1:' #
!
_user_specified_name	input_1
©
ў
4__inference_separable_conv2d_3_layer_call_fn_9445528

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_94455192
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
÷
S
'__inference_add_1_layer_call_fn_9446352
inputs_0
inputs_1
identity«
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_94456992
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:€€€€€€€€€@@А:€€€€€€€€€@@А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
љ
–
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_9445545

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐseparable_conv2d/ReadVariableOpҐ!separable_conv2d/ReadVariableOp_1і
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOpї
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateч
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
2
separable_conv2d/depthwiseф
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOp•
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
Seluа
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
љ
–
O__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_9445493

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐseparable_conv2d/ReadVariableOpҐ!separable_conv2d/ReadVariableOp_1і
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOpї
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateч
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
2
separable_conv2d/depthwiseф
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOp•
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
Seluа
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
ъ

ё
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9445604

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpґ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЫ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAdd∞
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Ј
ќ
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_9445421

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐseparable_conv2d/ReadVariableOpҐ!separable_conv2d/ReadVariableOp_1≥
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
separable_conv2d/ReadVariableOpЇ
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateц
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
separable_conv2d/depthwiseф
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOp•
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
Seluа
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ћ
Ђ
*__inference_conv2d_1_layer_call_fn_9445476

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*1
config_proto!

CPU

GPU2	 *0,1J 8*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_94454682
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
≥
f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_9445586

inputs
identityђ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
•[
џ
B__inference_model_layer_call_and_return_conditional_losses_9445759
input_10
,normalization_statefulpartitionedcall_args_10
,normalization_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_13
/separable_conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_35
1separable_conv2d_1_statefulpartitionedcall_args_15
1separable_conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_1_statefulpartitionedcall_args_3+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_15
1separable_conv2d_2_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_35
1separable_conv2d_3_statefulpartitionedcall_args_15
1separable_conv2d_3_statefulpartitionedcall_args_25
1separable_conv2d_3_statefulpartitionedcall_args_35
1separable_conv2d_4_statefulpartitionedcall_args_15
1separable_conv2d_4_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_35
1separable_conv2d_5_statefulpartitionedcall_args_15
1separable_conv2d_5_statefulpartitionedcall_args_25
1separable_conv2d_5_statefulpartitionedcall_args_3+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_24
0regression_head_1_statefulpartitionedcall_args_14
0regression_head_1_statefulpartitionedcall_args_2
identityИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ%normalization/StatefulPartitionedCallҐ)regression_head_1/StatefulPartitionedCallҐ(separable_conv2d/StatefulPartitionedCallҐ*separable_conv2d_1/StatefulPartitionedCallҐ*separable_conv2d_2/StatefulPartitionedCallҐ*separable_conv2d_3/StatefulPartitionedCallҐ*separable_conv2d_4/StatefulPartitionedCallҐ*separable_conv2d_5/StatefulPartitionedCall÷
%normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1,normalization_statefulpartitionedcall_args_1,normalization_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:€€€€€€€€€АА*1
config_proto!

CPU

GPU2	 *0,1J 8*S
fNRL
J__inference_normalization_layer_call_and_return_conditional_losses_94456442'
%normalization/StatefulPartitionedCallЎ
conv2d/StatefulPartitionedCallStatefulPartitionedCall.normalization/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€@@@*1
config_proto!

CPU

GPU2	 *0,1J 8*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_94453962 
conv2d/StatefulPartitionedCallґ
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0/separable_conv2d_statefulpartitionedcall_args_1/separable_conv2d_statefulpartitionedcall_args_2/separable_conv2d_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*V
fQRO
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_94454212*
(separable_conv2d/StatefulPartitionedCallћ
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:01separable_conv2d_1_statefulpartitionedcall_args_11separable_conv2d_1_statefulpartitionedcall_args_21separable_conv2d_1_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_94454472,
*separable_conv2d_1/StatefulPartitionedCall№
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_94454682"
 conv2d_1/StatefulPartitionedCallЩ
add/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_94456762
add/PartitionedCallЈ
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:01separable_conv2d_2_statefulpartitionedcall_args_11separable_conv2d_2_statefulpartitionedcall_args_21separable_conv2d_2_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_94454932,
*separable_conv2d_2/StatefulPartitionedCallќ
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:01separable_conv2d_3_statefulpartitionedcall_args_11separable_conv2d_3_statefulpartitionedcall_args_21separable_conv2d_3_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_94455192,
*separable_conv2d_3/StatefulPartitionedCallТ
add_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_94456992
add_1/PartitionedCallє
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:01separable_conv2d_4_statefulpartitionedcall_args_11separable_conv2d_4_statefulpartitionedcall_args_21separable_conv2d_4_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_94455452,
*separable_conv2d_4/StatefulPartitionedCallќ
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:01separable_conv2d_5_statefulpartitionedcall_args_11separable_conv2d_5_statefulpartitionedcall_args_21separable_conv2d_5_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_94455712,
*separable_conv2d_5/StatefulPartitionedCallЛ
max_pooling2d/PartitionedCallPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_94455862
max_pooling2d/PartitionedCall”
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_94456042"
 conv2d_2/StatefulPartitionedCallТ
add_2/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_94457262
add_2/PartitionedCallП
(global_average_pooling2d/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:€€€€€€€€€А*1
config_proto!

CPU

GPU2	 *0,1J 8*^
fYRW
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_94456192*
(global_average_pooling2d/PartitionedCallК
)regression_head_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:00regression_head_1_statefulpartitionedcall_args_10regression_head_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*W
fRRP
N__inference_regression_head_1_layer_call_and_return_conditional_losses_94457462+
)regression_head_1/StatefulPartitionedCallЌ
IdentityIdentity2regression_head_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall&^normalization/StatefulPartitionedCall*^regression_head_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ґ
_input_shapesР
Н:€€€€€€€€€АА::::::::::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2N
%normalization/StatefulPartitionedCall%normalization/StatefulPartitionedCall2V
)regression_head_1/StatefulPartitionedCall)regression_head_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
Ґ[
Џ
B__inference_model_layer_call_and_return_conditional_losses_9445939

inputs0
,normalization_statefulpartitionedcall_args_10
,normalization_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_13
/separable_conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_35
1separable_conv2d_1_statefulpartitionedcall_args_15
1separable_conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_1_statefulpartitionedcall_args_3+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_15
1separable_conv2d_2_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_35
1separable_conv2d_3_statefulpartitionedcall_args_15
1separable_conv2d_3_statefulpartitionedcall_args_25
1separable_conv2d_3_statefulpartitionedcall_args_35
1separable_conv2d_4_statefulpartitionedcall_args_15
1separable_conv2d_4_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_35
1separable_conv2d_5_statefulpartitionedcall_args_15
1separable_conv2d_5_statefulpartitionedcall_args_25
1separable_conv2d_5_statefulpartitionedcall_args_3+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_24
0regression_head_1_statefulpartitionedcall_args_14
0regression_head_1_statefulpartitionedcall_args_2
identityИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ%normalization/StatefulPartitionedCallҐ)regression_head_1/StatefulPartitionedCallҐ(separable_conv2d/StatefulPartitionedCallҐ*separable_conv2d_1/StatefulPartitionedCallҐ*separable_conv2d_2/StatefulPartitionedCallҐ*separable_conv2d_3/StatefulPartitionedCallҐ*separable_conv2d_4/StatefulPartitionedCallҐ*separable_conv2d_5/StatefulPartitionedCall’
%normalization/StatefulPartitionedCallStatefulPartitionedCallinputs,normalization_statefulpartitionedcall_args_1,normalization_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:€€€€€€€€€АА*1
config_proto!

CPU

GPU2	 *0,1J 8*S
fNRL
J__inference_normalization_layer_call_and_return_conditional_losses_94456442'
%normalization/StatefulPartitionedCallЎ
conv2d/StatefulPartitionedCallStatefulPartitionedCall.normalization/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€@@@*1
config_proto!

CPU

GPU2	 *0,1J 8*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_94453962 
conv2d/StatefulPartitionedCallґ
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0/separable_conv2d_statefulpartitionedcall_args_1/separable_conv2d_statefulpartitionedcall_args_2/separable_conv2d_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*V
fQRO
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_94454212*
(separable_conv2d/StatefulPartitionedCallћ
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:01separable_conv2d_1_statefulpartitionedcall_args_11separable_conv2d_1_statefulpartitionedcall_args_21separable_conv2d_1_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_94454472,
*separable_conv2d_1/StatefulPartitionedCall№
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_94454682"
 conv2d_1/StatefulPartitionedCallЩ
add/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_94456762
add/PartitionedCallЈ
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:01separable_conv2d_2_statefulpartitionedcall_args_11separable_conv2d_2_statefulpartitionedcall_args_21separable_conv2d_2_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_94454932,
*separable_conv2d_2/StatefulPartitionedCallќ
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:01separable_conv2d_3_statefulpartitionedcall_args_11separable_conv2d_3_statefulpartitionedcall_args_21separable_conv2d_3_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_94455192,
*separable_conv2d_3/StatefulPartitionedCallТ
add_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_94456992
add_1/PartitionedCallє
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:01separable_conv2d_4_statefulpartitionedcall_args_11separable_conv2d_4_statefulpartitionedcall_args_21separable_conv2d_4_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_94455452,
*separable_conv2d_4/StatefulPartitionedCallќ
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:01separable_conv2d_5_statefulpartitionedcall_args_11separable_conv2d_5_statefulpartitionedcall_args_21separable_conv2d_5_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_94455712,
*separable_conv2d_5/StatefulPartitionedCallЛ
max_pooling2d/PartitionedCallPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_94455862
max_pooling2d/PartitionedCall”
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_94456042"
 conv2d_2/StatefulPartitionedCallТ
add_2/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_94457262
add_2/PartitionedCallП
(global_average_pooling2d/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:€€€€€€€€€А*1
config_proto!

CPU

GPU2	 *0,1J 8*^
fYRW
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_94456192*
(global_average_pooling2d/PartitionedCallК
)regression_head_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:00regression_head_1_statefulpartitionedcall_args_10regression_head_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*W
fRRP
N__inference_regression_head_1_layer_call_and_return_conditional_losses_94457462+
)regression_head_1/StatefulPartitionedCallЌ
IdentityIdentity2regression_head_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall&^normalization/StatefulPartitionedCall*^regression_head_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ґ
_input_shapesР
Н:€€€€€€€€€АА::::::::::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2N
%normalization/StatefulPartitionedCall%normalization/StatefulPartitionedCall2V
)regression_head_1/StatefulPartitionedCall)regression_head_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
±
V
:__inference_global_average_pooling2d_layer_call_fn_9445625

inputs
identityЌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*^
fYRW
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_94456192
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
»†
•(
 __inference__traced_save_9446660
file_prefix1
-savev2_normalization_mean_read_readvariableop5
1savev2_normalization_variance_read_readvariableop2
.savev2_normalization_count_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop@
<savev2_separable_conv2d_depthwise_kernel_read_readvariableop@
<savev2_separable_conv2d_pointwise_kernel_read_readvariableop4
0savev2_separable_conv2d_bias_read_readvariableopB
>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableopB
>savev2_separable_conv2d_2_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_2_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_2_bias_read_readvariableopB
>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_3_bias_read_readvariableopB
>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_4_bias_read_readvariableopB
>savev2_separable_conv2d_5_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_5_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop7
3savev2_regression_head_1_kernel_read_readvariableop5
1savev2_regression_head_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop.
*savev2_conv2d_kernel_m_read_readvariableop,
(savev2_conv2d_bias_m_read_readvariableopB
>savev2_separable_conv2d_depthwise_kernel_m_read_readvariableopB
>savev2_separable_conv2d_pointwise_kernel_m_read_readvariableop6
2savev2_separable_conv2d_bias_m_read_readvariableopD
@savev2_separable_conv2d_1_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_1_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_1_bias_m_read_readvariableop0
,savev2_conv2d_1_kernel_m_read_readvariableop.
*savev2_conv2d_1_bias_m_read_readvariableopD
@savev2_separable_conv2d_2_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_2_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_2_bias_m_read_readvariableopD
@savev2_separable_conv2d_3_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_3_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_3_bias_m_read_readvariableopD
@savev2_separable_conv2d_4_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_4_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_4_bias_m_read_readvariableopD
@savev2_separable_conv2d_5_depthwise_kernel_m_read_readvariableopD
@savev2_separable_conv2d_5_pointwise_kernel_m_read_readvariableop8
4savev2_separable_conv2d_5_bias_m_read_readvariableop0
,savev2_conv2d_2_kernel_m_read_readvariableop.
*savev2_conv2d_2_bias_m_read_readvariableop9
5savev2_regression_head_1_kernel_m_read_readvariableop7
3savev2_regression_head_1_bias_m_read_readvariableop.
*savev2_conv2d_kernel_v_read_readvariableop,
(savev2_conv2d_bias_v_read_readvariableopB
>savev2_separable_conv2d_depthwise_kernel_v_read_readvariableopB
>savev2_separable_conv2d_pointwise_kernel_v_read_readvariableop6
2savev2_separable_conv2d_bias_v_read_readvariableopD
@savev2_separable_conv2d_1_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_1_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_1_bias_v_read_readvariableop0
,savev2_conv2d_1_kernel_v_read_readvariableop.
*savev2_conv2d_1_bias_v_read_readvariableopD
@savev2_separable_conv2d_2_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_2_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_2_bias_v_read_readvariableopD
@savev2_separable_conv2d_3_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_3_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_3_bias_v_read_readvariableopD
@savev2_separable_conv2d_4_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_4_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_4_bias_v_read_readvariableopD
@savev2_separable_conv2d_5_depthwise_kernel_v_read_readvariableopD
@savev2_separable_conv2d_5_pointwise_kernel_v_read_readvariableop8
4savev2_separable_conv2d_5_bias_v_read_readvariableop0
,savev2_conv2d_2_kernel_v_read_readvariableop.
*savev2_conv2d_2_bias_v_read_readvariableop9
5savev2_regression_head_1_kernel_v_read_readvariableop7
3savev2_regression_head_1_bias_v_read_readvariableop
savev2_1_const

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1•
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_41e995c1441643fea2c7588767625a88/part2
StringJoin/inputs_1Б

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameф3
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*Ж3
valueь2Bщ2UB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesµ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*њ
valueµB≤UB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesќ&
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_normalization_mean_read_readvariableop1savev2_normalization_variance_read_readvariableop.savev2_normalization_count_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop<savev2_separable_conv2d_depthwise_kernel_read_readvariableop<savev2_separable_conv2d_pointwise_kernel_read_readvariableop0savev2_separable_conv2d_bias_read_readvariableop>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_1_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop>savev2_separable_conv2d_2_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_2_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_2_bias_read_readvariableop>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_3_bias_read_readvariableop>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_4_bias_read_readvariableop>savev2_separable_conv2d_5_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_5_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_5_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop3savev2_regression_head_1_kernel_read_readvariableop1savev2_regression_head_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop*savev2_conv2d_kernel_m_read_readvariableop(savev2_conv2d_bias_m_read_readvariableop>savev2_separable_conv2d_depthwise_kernel_m_read_readvariableop>savev2_separable_conv2d_pointwise_kernel_m_read_readvariableop2savev2_separable_conv2d_bias_m_read_readvariableop@savev2_separable_conv2d_1_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_1_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_1_bias_m_read_readvariableop,savev2_conv2d_1_kernel_m_read_readvariableop*savev2_conv2d_1_bias_m_read_readvariableop@savev2_separable_conv2d_2_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_2_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_2_bias_m_read_readvariableop@savev2_separable_conv2d_3_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_3_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_3_bias_m_read_readvariableop@savev2_separable_conv2d_4_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_4_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_4_bias_m_read_readvariableop@savev2_separable_conv2d_5_depthwise_kernel_m_read_readvariableop@savev2_separable_conv2d_5_pointwise_kernel_m_read_readvariableop4savev2_separable_conv2d_5_bias_m_read_readvariableop,savev2_conv2d_2_kernel_m_read_readvariableop*savev2_conv2d_2_bias_m_read_readvariableop5savev2_regression_head_1_kernel_m_read_readvariableop3savev2_regression_head_1_bias_m_read_readvariableop*savev2_conv2d_kernel_v_read_readvariableop(savev2_conv2d_bias_v_read_readvariableop>savev2_separable_conv2d_depthwise_kernel_v_read_readvariableop>savev2_separable_conv2d_pointwise_kernel_v_read_readvariableop2savev2_separable_conv2d_bias_v_read_readvariableop@savev2_separable_conv2d_1_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_1_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_1_bias_v_read_readvariableop,savev2_conv2d_1_kernel_v_read_readvariableop*savev2_conv2d_1_bias_v_read_readvariableop@savev2_separable_conv2d_2_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_2_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_2_bias_v_read_readvariableop@savev2_separable_conv2d_3_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_3_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_3_bias_v_read_readvariableop@savev2_separable_conv2d_4_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_4_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_4_bias_v_read_readvariableop@savev2_separable_conv2d_5_depthwise_kernel_v_read_readvariableop@savev2_separable_conv2d_5_pointwise_kernel_v_read_readvariableop4savev2_separable_conv2d_5_bias_v_read_readvariableop,savev2_conv2d_2_kernel_v_read_readvariableop*savev2_conv2d_2_bias_v_read_readvariableop5savev2_regression_head_1_kernel_v_read_readvariableop3savev2_regression_head_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *c
dtypesY
W2U2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardђ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ґ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesѕ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1г
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesђ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*€
_input_shapesн
к: ::: :@:@:@:@А:А:А:АА:А:@А:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:АА:А:	А:: : : : :@:@:@:@А:А:А:АА:А:@А:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:АА:А:	А::@:@:@:@А:А:А:АА:А:@А:А:А:АА:А:А:АА:А:А:АА:А:А:АА:А:АА:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
Т
г
'__inference_model_layer_call_fn_9446306

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28
identityИҐStatefulPartitionedCallш	
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28*(
Tin!
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_94459392
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ґ
_input_shapesР
Н:€€€€€€€€€АА::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
•[
џ
B__inference_model_layer_call_and_return_conditional_losses_9445807
input_10
,normalization_statefulpartitionedcall_args_10
,normalization_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_13
/separable_conv2d_statefulpartitionedcall_args_23
/separable_conv2d_statefulpartitionedcall_args_35
1separable_conv2d_1_statefulpartitionedcall_args_15
1separable_conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_1_statefulpartitionedcall_args_3+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_15
1separable_conv2d_2_statefulpartitionedcall_args_25
1separable_conv2d_2_statefulpartitionedcall_args_35
1separable_conv2d_3_statefulpartitionedcall_args_15
1separable_conv2d_3_statefulpartitionedcall_args_25
1separable_conv2d_3_statefulpartitionedcall_args_35
1separable_conv2d_4_statefulpartitionedcall_args_15
1separable_conv2d_4_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_35
1separable_conv2d_5_statefulpartitionedcall_args_15
1separable_conv2d_5_statefulpartitionedcall_args_25
1separable_conv2d_5_statefulpartitionedcall_args_3+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_24
0regression_head_1_statefulpartitionedcall_args_14
0regression_head_1_statefulpartitionedcall_args_2
identityИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ%normalization/StatefulPartitionedCallҐ)regression_head_1/StatefulPartitionedCallҐ(separable_conv2d/StatefulPartitionedCallҐ*separable_conv2d_1/StatefulPartitionedCallҐ*separable_conv2d_2/StatefulPartitionedCallҐ*separable_conv2d_3/StatefulPartitionedCallҐ*separable_conv2d_4/StatefulPartitionedCallҐ*separable_conv2d_5/StatefulPartitionedCall÷
%normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1,normalization_statefulpartitionedcall_args_1,normalization_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:€€€€€€€€€АА*1
config_proto!

CPU

GPU2	 *0,1J 8*S
fNRL
J__inference_normalization_layer_call_and_return_conditional_losses_94456442'
%normalization/StatefulPartitionedCallЎ
conv2d/StatefulPartitionedCallStatefulPartitionedCall.normalization/StatefulPartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€@@@*1
config_proto!

CPU

GPU2	 *0,1J 8*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_94453962 
conv2d/StatefulPartitionedCallґ
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0/separable_conv2d_statefulpartitionedcall_args_1/separable_conv2d_statefulpartitionedcall_args_2/separable_conv2d_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*V
fQRO
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_94454212*
(separable_conv2d/StatefulPartitionedCallћ
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:01separable_conv2d_1_statefulpartitionedcall_args_11separable_conv2d_1_statefulpartitionedcall_args_21separable_conv2d_1_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_94454472,
*separable_conv2d_1/StatefulPartitionedCall№
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_94454682"
 conv2d_1/StatefulPartitionedCallЩ
add/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_94456762
add/PartitionedCallЈ
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:01separable_conv2d_2_statefulpartitionedcall_args_11separable_conv2d_2_statefulpartitionedcall_args_21separable_conv2d_2_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_94454932,
*separable_conv2d_2/StatefulPartitionedCallќ
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:01separable_conv2d_3_statefulpartitionedcall_args_11separable_conv2d_3_statefulpartitionedcall_args_21separable_conv2d_3_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_94455192,
*separable_conv2d_3/StatefulPartitionedCallТ
add_1/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_94456992
add_1/PartitionedCallє
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:01separable_conv2d_4_statefulpartitionedcall_args_11separable_conv2d_4_statefulpartitionedcall_args_21separable_conv2d_4_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_94455452,
*separable_conv2d_4/StatefulPartitionedCallќ
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:01separable_conv2d_5_statefulpartitionedcall_args_11separable_conv2d_5_statefulpartitionedcall_args_21separable_conv2d_5_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€@@А*1
config_proto!

CPU

GPU2	 *0,1J 8*X
fSRQ
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_94455712,
*separable_conv2d_5/StatefulPartitionedCallЛ
max_pooling2d/PartitionedCallPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_94455862
max_pooling2d/PartitionedCall”
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_94456042"
 conv2d_2/StatefulPartitionedCallТ
add_2/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_94457262
add_2/PartitionedCallП
(global_average_pooling2d/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:€€€€€€€€€А*1
config_proto!

CPU

GPU2	 *0,1J 8*^
fYRW
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_94456192*
(global_average_pooling2d/PartitionedCallК
)regression_head_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:00regression_head_1_statefulpartitionedcall_args_10regression_head_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*W
fRRP
N__inference_regression_head_1_layer_call_and_return_conditional_losses_94457462+
)regression_head_1/StatefulPartitionedCallЌ
IdentityIdentity2regression_head_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall&^normalization/StatefulPartitionedCall*^regression_head_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ґ
_input_shapesР
Н:€€€€€€€€€АА::::::::::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2N
%normalization/StatefulPartitionedCall%normalization/StatefulPartitionedCall2V
)regression_head_1/StatefulPartitionedCall)regression_head_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
з
№
C__inference_conv2d_layer_call_and_return_conditional_losses_9445396

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
Selu±
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Т
г
'__inference_model_layer_call_fn_9446273

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28
identityИҐStatefulPartitionedCallш	
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28*(
Tin!
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_94458582
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ґ
_input_shapesР
Н:€€€€€€€€€АА::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
–
K
/__inference_max_pooling2d_layer_call_fn_9445592

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*1
config_proto!

CPU

GPU2	 *0,1J 8*S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_94455862
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:& "
 
_user_specified_nameinputs
џ
й
J__inference_normalization_layer_call_and_return_conditional_losses_9446321

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identityИҐReshape/ReadVariableOpҐReshape_1/ReadVariableOpМ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shapeЖ
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
ReshapeТ
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shapeО
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1g
subSubinputsReshape:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
subY
SqrtSqrtReshape_1:output:0*
T0*&
_output_shapes
:2
Sqrtl
truedivRealDivsub:z:0Sqrt:y:0*
T0*1
_output_shapes
:€€€€€€€€€АА2	
truedivЭ
IdentityIdentitytruediv:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:& "
 
_user_specified_nameinputs
£
∞
/__inference_normalization_layer_call_fn_9446328

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:€€€€€€€€€АА*1
config_proto!

CPU

GPU2	 *0,1J 8*S
fNRL
J__inference_normalization_layer_call_and_return_conditional_losses_94456442
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
љ
–
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_9445571

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐseparable_conv2d/ReadVariableOpҐ!separable_conv2d/ReadVariableOp_1і
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02!
separable_conv2d/ReadVariableOpї
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype02#
!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/ShapeС
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateч
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
2
separable_conv2d/depthwiseф
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingVALID*
strides
2
separable_conv2dН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOp•
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAdds
SeluSeluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
Seluа
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
ш

ё
E__inference_conv2d_1_layer_call_and_return_conditional_losses_9445468

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOpґ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЫ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAdd∞
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
≈
©
(__inference_conv2d_layer_call_fn_9445404

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*1
config_proto!

CPU

GPU2	 *0,1J 8*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_94453962
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ив
й1
#__inference__traced_restore_9446927
file_prefix'
#assignvariableop_normalization_mean-
)assignvariableop_1_normalization_variance*
&assignvariableop_2_normalization_count$
 assignvariableop_3_conv2d_kernel"
assignvariableop_4_conv2d_bias8
4assignvariableop_5_separable_conv2d_depthwise_kernel8
4assignvariableop_6_separable_conv2d_pointwise_kernel,
(assignvariableop_7_separable_conv2d_bias:
6assignvariableop_8_separable_conv2d_1_depthwise_kernel:
6assignvariableop_9_separable_conv2d_1_pointwise_kernel/
+assignvariableop_10_separable_conv2d_1_bias'
#assignvariableop_11_conv2d_1_kernel%
!assignvariableop_12_conv2d_1_bias;
7assignvariableop_13_separable_conv2d_2_depthwise_kernel;
7assignvariableop_14_separable_conv2d_2_pointwise_kernel/
+assignvariableop_15_separable_conv2d_2_bias;
7assignvariableop_16_separable_conv2d_3_depthwise_kernel;
7assignvariableop_17_separable_conv2d_3_pointwise_kernel/
+assignvariableop_18_separable_conv2d_3_bias;
7assignvariableop_19_separable_conv2d_4_depthwise_kernel;
7assignvariableop_20_separable_conv2d_4_pointwise_kernel/
+assignvariableop_21_separable_conv2d_4_bias;
7assignvariableop_22_separable_conv2d_5_depthwise_kernel;
7assignvariableop_23_separable_conv2d_5_pointwise_kernel/
+assignvariableop_24_separable_conv2d_5_bias'
#assignvariableop_25_conv2d_2_kernel%
!assignvariableop_26_conv2d_2_bias0
,assignvariableop_27_regression_head_1_kernel.
*assignvariableop_28_regression_head_1_bias
assignvariableop_29_total
assignvariableop_30_count
assignvariableop_31_total_1
assignvariableop_32_count_1'
#assignvariableop_33_conv2d_kernel_m%
!assignvariableop_34_conv2d_bias_m;
7assignvariableop_35_separable_conv2d_depthwise_kernel_m;
7assignvariableop_36_separable_conv2d_pointwise_kernel_m/
+assignvariableop_37_separable_conv2d_bias_m=
9assignvariableop_38_separable_conv2d_1_depthwise_kernel_m=
9assignvariableop_39_separable_conv2d_1_pointwise_kernel_m1
-assignvariableop_40_separable_conv2d_1_bias_m)
%assignvariableop_41_conv2d_1_kernel_m'
#assignvariableop_42_conv2d_1_bias_m=
9assignvariableop_43_separable_conv2d_2_depthwise_kernel_m=
9assignvariableop_44_separable_conv2d_2_pointwise_kernel_m1
-assignvariableop_45_separable_conv2d_2_bias_m=
9assignvariableop_46_separable_conv2d_3_depthwise_kernel_m=
9assignvariableop_47_separable_conv2d_3_pointwise_kernel_m1
-assignvariableop_48_separable_conv2d_3_bias_m=
9assignvariableop_49_separable_conv2d_4_depthwise_kernel_m=
9assignvariableop_50_separable_conv2d_4_pointwise_kernel_m1
-assignvariableop_51_separable_conv2d_4_bias_m=
9assignvariableop_52_separable_conv2d_5_depthwise_kernel_m=
9assignvariableop_53_separable_conv2d_5_pointwise_kernel_m1
-assignvariableop_54_separable_conv2d_5_bias_m)
%assignvariableop_55_conv2d_2_kernel_m'
#assignvariableop_56_conv2d_2_bias_m2
.assignvariableop_57_regression_head_1_kernel_m0
,assignvariableop_58_regression_head_1_bias_m'
#assignvariableop_59_conv2d_kernel_v%
!assignvariableop_60_conv2d_bias_v;
7assignvariableop_61_separable_conv2d_depthwise_kernel_v;
7assignvariableop_62_separable_conv2d_pointwise_kernel_v/
+assignvariableop_63_separable_conv2d_bias_v=
9assignvariableop_64_separable_conv2d_1_depthwise_kernel_v=
9assignvariableop_65_separable_conv2d_1_pointwise_kernel_v1
-assignvariableop_66_separable_conv2d_1_bias_v)
%assignvariableop_67_conv2d_1_kernel_v'
#assignvariableop_68_conv2d_1_bias_v=
9assignvariableop_69_separable_conv2d_2_depthwise_kernel_v=
9assignvariableop_70_separable_conv2d_2_pointwise_kernel_v1
-assignvariableop_71_separable_conv2d_2_bias_v=
9assignvariableop_72_separable_conv2d_3_depthwise_kernel_v=
9assignvariableop_73_separable_conv2d_3_pointwise_kernel_v1
-assignvariableop_74_separable_conv2d_3_bias_v=
9assignvariableop_75_separable_conv2d_4_depthwise_kernel_v=
9assignvariableop_76_separable_conv2d_4_pointwise_kernel_v1
-assignvariableop_77_separable_conv2d_4_bias_v=
9assignvariableop_78_separable_conv2d_5_depthwise_kernel_v=
9assignvariableop_79_separable_conv2d_5_pointwise_kernel_v1
-assignvariableop_80_separable_conv2d_5_bias_v)
%assignvariableop_81_conv2d_2_kernel_v'
#assignvariableop_82_conv2d_2_bias_v2
.assignvariableop_83_regression_head_1_kernel_v0
,assignvariableop_84_regression_head_1_bias_v
identity_86ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1ъ3
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*Ж3
valueь2Bщ2UB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesї
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*њ
valueµB≤UB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices„
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*к
_output_shapes„
‘:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*c
dtypesY
W2U2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityУ
AssignVariableOpAssignVariableOp#assignvariableop_normalization_meanIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Я
AssignVariableOp_1AssignVariableOp)assignvariableop_1_normalization_varianceIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ь
AssignVariableOp_2AssignVariableOp&assignvariableop_2_normalization_countIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ц
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ф
AssignVariableOp_4AssignVariableOpassignvariableop_4_conv2d_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5™
AssignVariableOp_5AssignVariableOp4assignvariableop_5_separable_conv2d_depthwise_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6™
AssignVariableOp_6AssignVariableOp4assignvariableop_6_separable_conv2d_pointwise_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ю
AssignVariableOp_7AssignVariableOp(assignvariableop_7_separable_conv2d_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8ђ
AssignVariableOp_8AssignVariableOp6assignvariableop_8_separable_conv2d_1_depthwise_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9ђ
AssignVariableOp_9AssignVariableOp6assignvariableop_9_separable_conv2d_1_pointwise_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10§
AssignVariableOp_10AssignVariableOp+assignvariableop_10_separable_conv2d_1_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11Ь
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_1_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ъ
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv2d_1_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13∞
AssignVariableOp_13AssignVariableOp7assignvariableop_13_separable_conv2d_2_depthwise_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14∞
AssignVariableOp_14AssignVariableOp7assignvariableop_14_separable_conv2d_2_pointwise_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15§
AssignVariableOp_15AssignVariableOp+assignvariableop_15_separable_conv2d_2_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16∞
AssignVariableOp_16AssignVariableOp7assignvariableop_16_separable_conv2d_3_depthwise_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17∞
AssignVariableOp_17AssignVariableOp7assignvariableop_17_separable_conv2d_3_pointwise_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18§
AssignVariableOp_18AssignVariableOp+assignvariableop_18_separable_conv2d_3_biasIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19∞
AssignVariableOp_19AssignVariableOp7assignvariableop_19_separable_conv2d_4_depthwise_kernelIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20∞
AssignVariableOp_20AssignVariableOp7assignvariableop_20_separable_conv2d_4_pointwise_kernelIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21§
AssignVariableOp_21AssignVariableOp+assignvariableop_21_separable_conv2d_4_biasIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22∞
AssignVariableOp_22AssignVariableOp7assignvariableop_22_separable_conv2d_5_depthwise_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23∞
AssignVariableOp_23AssignVariableOp7assignvariableop_23_separable_conv2d_5_pointwise_kernelIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24§
AssignVariableOp_24AssignVariableOp+assignvariableop_24_separable_conv2d_5_biasIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Ь
AssignVariableOp_25AssignVariableOp#assignvariableop_25_conv2d_2_kernelIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Ъ
AssignVariableOp_26AssignVariableOp!assignvariableop_26_conv2d_2_biasIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27•
AssignVariableOp_27AssignVariableOp,assignvariableop_27_regression_head_1_kernelIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28£
AssignVariableOp_28AssignVariableOp*assignvariableop_28_regression_head_1_biasIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29Т
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30Т
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Ф
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_1Identity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Ф
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_1Identity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Ь
AssignVariableOp_33AssignVariableOp#assignvariableop_33_conv2d_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34Ъ
AssignVariableOp_34AssignVariableOp!assignvariableop_34_conv2d_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35∞
AssignVariableOp_35AssignVariableOp7assignvariableop_35_separable_conv2d_depthwise_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36∞
AssignVariableOp_36AssignVariableOp7assignvariableop_36_separable_conv2d_pointwise_kernel_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37§
AssignVariableOp_37AssignVariableOp+assignvariableop_37_separable_conv2d_bias_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38≤
AssignVariableOp_38AssignVariableOp9assignvariableop_38_separable_conv2d_1_depthwise_kernel_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39≤
AssignVariableOp_39AssignVariableOp9assignvariableop_39_separable_conv2d_1_pointwise_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40¶
AssignVariableOp_40AssignVariableOp-assignvariableop_40_separable_conv2d_1_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41Ю
AssignVariableOp_41AssignVariableOp%assignvariableop_41_conv2d_1_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42Ь
AssignVariableOp_42AssignVariableOp#assignvariableop_42_conv2d_1_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43≤
AssignVariableOp_43AssignVariableOp9assignvariableop_43_separable_conv2d_2_depthwise_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44≤
AssignVariableOp_44AssignVariableOp9assignvariableop_44_separable_conv2d_2_pointwise_kernel_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45¶
AssignVariableOp_45AssignVariableOp-assignvariableop_45_separable_conv2d_2_bias_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46≤
AssignVariableOp_46AssignVariableOp9assignvariableop_46_separable_conv2d_3_depthwise_kernel_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47≤
AssignVariableOp_47AssignVariableOp9assignvariableop_47_separable_conv2d_3_pointwise_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48¶
AssignVariableOp_48AssignVariableOp-assignvariableop_48_separable_conv2d_3_bias_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49≤
AssignVariableOp_49AssignVariableOp9assignvariableop_49_separable_conv2d_4_depthwise_kernel_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50≤
AssignVariableOp_50AssignVariableOp9assignvariableop_50_separable_conv2d_4_pointwise_kernel_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51¶
AssignVariableOp_51AssignVariableOp-assignvariableop_51_separable_conv2d_4_bias_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52≤
AssignVariableOp_52AssignVariableOp9assignvariableop_52_separable_conv2d_5_depthwise_kernel_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53≤
AssignVariableOp_53AssignVariableOp9assignvariableop_53_separable_conv2d_5_pointwise_kernel_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54¶
AssignVariableOp_54AssignVariableOp-assignvariableop_54_separable_conv2d_5_bias_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55Ю
AssignVariableOp_55AssignVariableOp%assignvariableop_55_conv2d_2_kernel_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56Ь
AssignVariableOp_56AssignVariableOp#assignvariableop_56_conv2d_2_bias_mIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57І
AssignVariableOp_57AssignVariableOp.assignvariableop_57_regression_head_1_kernel_mIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58•
AssignVariableOp_58AssignVariableOp,assignvariableop_58_regression_head_1_bias_mIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59Ь
AssignVariableOp_59AssignVariableOp#assignvariableop_59_conv2d_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60Ъ
AssignVariableOp_60AssignVariableOp!assignvariableop_60_conv2d_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61∞
AssignVariableOp_61AssignVariableOp7assignvariableop_61_separable_conv2d_depthwise_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62∞
AssignVariableOp_62AssignVariableOp7assignvariableop_62_separable_conv2d_pointwise_kernel_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63§
AssignVariableOp_63AssignVariableOp+assignvariableop_63_separable_conv2d_bias_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64≤
AssignVariableOp_64AssignVariableOp9assignvariableop_64_separable_conv2d_1_depthwise_kernel_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65≤
AssignVariableOp_65AssignVariableOp9assignvariableop_65_separable_conv2d_1_pointwise_kernel_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66¶
AssignVariableOp_66AssignVariableOp-assignvariableop_66_separable_conv2d_1_bias_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67Ю
AssignVariableOp_67AssignVariableOp%assignvariableop_67_conv2d_1_kernel_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68Ь
AssignVariableOp_68AssignVariableOp#assignvariableop_68_conv2d_1_bias_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69≤
AssignVariableOp_69AssignVariableOp9assignvariableop_69_separable_conv2d_2_depthwise_kernel_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70≤
AssignVariableOp_70AssignVariableOp9assignvariableop_70_separable_conv2d_2_pointwise_kernel_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71¶
AssignVariableOp_71AssignVariableOp-assignvariableop_71_separable_conv2d_2_bias_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72≤
AssignVariableOp_72AssignVariableOp9assignvariableop_72_separable_conv2d_3_depthwise_kernel_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73≤
AssignVariableOp_73AssignVariableOp9assignvariableop_73_separable_conv2d_3_pointwise_kernel_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74¶
AssignVariableOp_74AssignVariableOp-assignvariableop_74_separable_conv2d_3_bias_vIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75≤
AssignVariableOp_75AssignVariableOp9assignvariableop_75_separable_conv2d_4_depthwise_kernel_vIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76≤
AssignVariableOp_76AssignVariableOp9assignvariableop_76_separable_conv2d_4_pointwise_kernel_vIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
T0*
_output_shapes
:2
Identity_77¶
AssignVariableOp_77AssignVariableOp-assignvariableop_77_separable_conv2d_4_bias_vIdentity_77:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:2
Identity_78≤
AssignVariableOp_78AssignVariableOp9assignvariableop_78_separable_conv2d_5_depthwise_kernel_vIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78_
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:2
Identity_79≤
AssignVariableOp_79AssignVariableOp9assignvariableop_79_separable_conv2d_5_pointwise_kernel_vIdentity_79:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_79_
Identity_80IdentityRestoreV2:tensors:80*
T0*
_output_shapes
:2
Identity_80¶
AssignVariableOp_80AssignVariableOp-assignvariableop_80_separable_conv2d_5_bias_vIdentity_80:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_80_
Identity_81IdentityRestoreV2:tensors:81*
T0*
_output_shapes
:2
Identity_81Ю
AssignVariableOp_81AssignVariableOp%assignvariableop_81_conv2d_2_kernel_vIdentity_81:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_81_
Identity_82IdentityRestoreV2:tensors:82*
T0*
_output_shapes
:2
Identity_82Ь
AssignVariableOp_82AssignVariableOp#assignvariableop_82_conv2d_2_bias_vIdentity_82:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_82_
Identity_83IdentityRestoreV2:tensors:83*
T0*
_output_shapes
:2
Identity_83І
AssignVariableOp_83AssignVariableOp.assignvariableop_83_regression_head_1_kernel_vIdentity_83:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_83_
Identity_84IdentityRestoreV2:tensors:84*
T0*
_output_shapes
:2
Identity_84•
AssignVariableOp_84AssignVariableOp,assignvariableop_84_regression_head_1_bias_vIdentity_84:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_84®
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesƒ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpђ
Identity_85Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_85є
Identity_86IdentityIdentity_85:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_86"#
identity_86Identity_86:output:0*л
_input_shapesў
÷: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
ћ√
Щ
B__inference_model_layer_call_and_return_conditional_losses_9446240

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource=
9separable_conv2d_separable_conv2d_readvariableop_resource?
;separable_conv2d_separable_conv2d_readvariableop_1_resource4
0separable_conv2d_biasadd_readvariableop_resource?
;separable_conv2d_1_separable_conv2d_readvariableop_resourceA
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_1_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource?
;separable_conv2d_2_separable_conv2d_readvariableop_resourceA
=separable_conv2d_2_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_2_biasadd_readvariableop_resource?
;separable_conv2d_3_separable_conv2d_readvariableop_resourceA
=separable_conv2d_3_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_3_biasadd_readvariableop_resource?
;separable_conv2d_4_separable_conv2d_readvariableop_resourceA
=separable_conv2d_4_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_4_biasadd_readvariableop_resource?
;separable_conv2d_5_separable_conv2d_readvariableop_resourceA
=separable_conv2d_5_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_5_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource4
0regression_head_1_matmul_readvariableop_resource5
1regression_head_1_biasadd_readvariableop_resource
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐ$normalization/Reshape/ReadVariableOpҐ&normalization/Reshape_1/ReadVariableOpҐ(regression_head_1/BiasAdd/ReadVariableOpҐ'regression_head_1/MatMul/ReadVariableOpҐ'separable_conv2d/BiasAdd/ReadVariableOpҐ0separable_conv2d/separable_conv2d/ReadVariableOpҐ2separable_conv2d/separable_conv2d/ReadVariableOp_1Ґ)separable_conv2d_1/BiasAdd/ReadVariableOpҐ2separable_conv2d_1/separable_conv2d/ReadVariableOpҐ4separable_conv2d_1/separable_conv2d/ReadVariableOp_1Ґ)separable_conv2d_2/BiasAdd/ReadVariableOpҐ2separable_conv2d_2/separable_conv2d/ReadVariableOpҐ4separable_conv2d_2/separable_conv2d/ReadVariableOp_1Ґ)separable_conv2d_3/BiasAdd/ReadVariableOpҐ2separable_conv2d_3/separable_conv2d/ReadVariableOpҐ4separable_conv2d_3/separable_conv2d/ReadVariableOp_1Ґ)separable_conv2d_4/BiasAdd/ReadVariableOpҐ2separable_conv2d_4/separable_conv2d/ReadVariableOpҐ4separable_conv2d_4/separable_conv2d/ReadVariableOp_1Ґ)separable_conv2d_5/BiasAdd/ReadVariableOpҐ2separable_conv2d_5/separable_conv2d/ReadVariableOpҐ4separable_conv2d_5/separable_conv2d/ReadVariableOp_1ґ
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOpУ
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shapeЊ
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/ReshapeЉ
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOpЧ
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shape∆
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1С
normalization/subSubinputsnormalization/Reshape:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
normalization/subГ
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrt§
normalization/truedivRealDivnormalization/sub:z:0normalization/Sqrt:y:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
normalization/truediv™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOpЋ
conv2d/Conv2DConv2Dnormalization/truediv:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@*
paddingSAME*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2
conv2d/BiasAddu
conv2d/SeluSeluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2
conv2d/Seluж
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype022
0separable_conv2d/separable_conv2d/ReadVariableOpн
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@А*
dtype024
2separable_conv2d/separable_conv2d/ReadVariableOp_1Ђ
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2)
'separable_conv2d/separable_conv2d/Shape≥
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      21
/separable_conv2d/separable_conv2d/dilation_rate™
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeconv2d/Selu:activations:08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@*
paddingSAME*
strides
2-
+separable_conv2d/separable_conv2d/depthwise¶
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2#
!separable_conv2d/separable_conv2dј
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02)
'separable_conv2d/BiasAdd/ReadVariableOp„
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d/BiasAddФ
separable_conv2d/SeluSelu!separable_conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d/Seluн
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_1/separable_conv2d/ReadVariableOpф
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ѓ
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2+
)separable_conv2d_1/separable_conv2d/ShapeЈ
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_1/separable_conv2d/dilation_rateї
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Selu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2/
-separable_conv2d_1/separable_conv2d/depthwiseЃ
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2%
#separable_conv2d_1/separable_conv2d∆
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_1/BiasAdd/ReadVariableOpя
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_1/BiasAddЪ
separable_conv2d_1/SeluSelu#separable_conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_1/Selu±
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_1/Conv2D/ReadVariableOp“
conv2d_1/Conv2DConv2Dconv2d/Selu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2
conv2d_1/Conv2D®
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp≠
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
conv2d_1/BiasAddШ
add/addAddV2%separable_conv2d_1/Selu:activations:0conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2	
add/addн
2separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_2_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_2/separable_conv2d/ReadVariableOpф
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_2_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ѓ
)separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2+
)separable_conv2d_2/separable_conv2d/ShapeЈ
1separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_2/separable_conv2d/dilation_rate£
-separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativeadd/add:z:0:separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2/
-separable_conv2d_2/separable_conv2d/depthwiseЃ
#separable_conv2d_2/separable_conv2dConv2D6separable_conv2d_2/separable_conv2d/depthwise:output:0<separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2%
#separable_conv2d_2/separable_conv2d∆
)separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_2/BiasAdd/ReadVariableOpя
separable_conv2d_2/BiasAddBiasAdd,separable_conv2d_2/separable_conv2d:output:01separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_2/BiasAddЪ
separable_conv2d_2/SeluSelu#separable_conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_2/Seluн
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_3/separable_conv2d/ReadVariableOpф
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ѓ
)separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2+
)separable_conv2d_3/separable_conv2d/ShapeЈ
1separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_3/separable_conv2d/dilation_rateљ
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_2/Selu:activations:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2/
-separable_conv2d_3/separable_conv2d/depthwiseЃ
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2%
#separable_conv2d_3/separable_conv2d∆
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_3/BiasAdd/ReadVariableOpя
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_3/BiasAddЪ
separable_conv2d_3/SeluSelu#separable_conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_3/SeluО
	add_1/addAddV2%separable_conv2d_3/Selu:activations:0add/add:z:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
	add_1/addн
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_4/separable_conv2d/ReadVariableOpф
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ѓ
)separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2+
)separable_conv2d_4/separable_conv2d/ShapeЈ
1separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_4/separable_conv2d/dilation_rate•
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNativeadd_1/add:z:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2/
-separable_conv2d_4/separable_conv2d/depthwiseЃ
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2%
#separable_conv2d_4/separable_conv2d∆
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_4/BiasAdd/ReadVariableOpя
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_4/BiasAddЪ
separable_conv2d_4/SeluSelu#separable_conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_4/Seluн
2separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_5_separable_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype024
2separable_conv2d_5/separable_conv2d/ReadVariableOpф
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_5_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:АА*
dtype026
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1ѓ
)separable_conv2d_5/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_5/separable_conv2d/ShapeЈ
1separable_conv2d_5/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_5/separable_conv2d/dilation_rateљ
-separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_4/Selu:activations:0:separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingSAME*
strides
2/
-separable_conv2d_5/separable_conv2d/depthwiseЃ
#separable_conv2d_5/separable_conv2dConv2D6separable_conv2d_5/separable_conv2d/depthwise:output:0<separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А*
paddingVALID*
strides
2%
#separable_conv2d_5/separable_conv2d∆
)separable_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)separable_conv2d_5/BiasAdd/ReadVariableOpя
separable_conv2d_5/BiasAddBiasAdd,separable_conv2d_5/separable_conv2d:output:01separable_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_5/BiasAddЪ
separable_conv2d_5/SeluSelu#separable_conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2
separable_conv2d_5/SeluЌ
max_pooling2d/MaxPoolMaxPool%separable_conv2d_5/Selu:activations:0*0
_output_shapes
:€€€€€€€€€  А*
ksize
*
paddingSAME*
strides
2
max_pooling2d/MaxPool≤
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02 
conv2d_2/Conv2D/ReadVariableOp∆
conv2d_2/Conv2DConv2Dadd_1/add:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€  А*
paddingSAME*
strides
2
conv2d_2/Conv2D®
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp≠
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€  А2
conv2d_2/BiasAddХ
	add_2/addAddV2max_pooling2d/MaxPool:output:0conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€  А2
	add_2/add≥
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices¬
global_average_pooling2d/MeanMeanadd_2/add:z:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
global_average_pooling2d/Meanƒ
'regression_head_1/MatMul/ReadVariableOpReadVariableOp0regression_head_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02)
'regression_head_1/MatMul/ReadVariableOp…
regression_head_1/MatMulMatMul&global_average_pooling2d/Mean:output:0/regression_head_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
regression_head_1/MatMul¬
(regression_head_1/BiasAdd/ReadVariableOpReadVariableOp1regression_head_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(regression_head_1/BiasAdd/ReadVariableOp…
regression_head_1/BiasAddBiasAdd"regression_head_1/MatMul:product:00regression_head_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
regression_head_1/BiasAddк

IdentityIdentity"regression_head_1/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp)^regression_head_1/BiasAdd/ReadVariableOp(^regression_head_1/MatMul/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*^separable_conv2d_2/BiasAdd/ReadVariableOp3^separable_conv2d_2/separable_conv2d/ReadVariableOp5^separable_conv2d_2/separable_conv2d/ReadVariableOp_1*^separable_conv2d_3/BiasAdd/ReadVariableOp3^separable_conv2d_3/separable_conv2d/ReadVariableOp5^separable_conv2d_3/separable_conv2d/ReadVariableOp_1*^separable_conv2d_4/BiasAdd/ReadVariableOp3^separable_conv2d_4/separable_conv2d/ReadVariableOp5^separable_conv2d_4/separable_conv2d/ReadVariableOp_1*^separable_conv2d_5/BiasAdd/ReadVariableOp3^separable_conv2d_5/separable_conv2d/ReadVariableOp5^separable_conv2d_5/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ґ
_input_shapesР
Н:€€€€€€€€€АА::::::::::::::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2T
(regression_head_1/BiasAdd/ReadVariableOp(regression_head_1/BiasAdd/ReadVariableOp2R
'regression_head_1/MatMul/ReadVariableOp'regression_head_1/MatMul/ReadVariableOp2R
'separable_conv2d/BiasAdd/ReadVariableOp'separable_conv2d/BiasAdd/ReadVariableOp2d
0separable_conv2d/separable_conv2d/ReadVariableOp0separable_conv2d/separable_conv2d/ReadVariableOp2h
2separable_conv2d/separable_conv2d/ReadVariableOp_12separable_conv2d/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_1/BiasAdd/ReadVariableOp)separable_conv2d_1/BiasAdd/ReadVariableOp2h
2separable_conv2d_1/separable_conv2d/ReadVariableOp2separable_conv2d_1/separable_conv2d/ReadVariableOp2l
4separable_conv2d_1/separable_conv2d/ReadVariableOp_14separable_conv2d_1/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_2/BiasAdd/ReadVariableOp)separable_conv2d_2/BiasAdd/ReadVariableOp2h
2separable_conv2d_2/separable_conv2d/ReadVariableOp2separable_conv2d_2/separable_conv2d/ReadVariableOp2l
4separable_conv2d_2/separable_conv2d/ReadVariableOp_14separable_conv2d_2/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_3/BiasAdd/ReadVariableOp)separable_conv2d_3/BiasAdd/ReadVariableOp2h
2separable_conv2d_3/separable_conv2d/ReadVariableOp2separable_conv2d_3/separable_conv2d/ReadVariableOp2l
4separable_conv2d_3/separable_conv2d/ReadVariableOp_14separable_conv2d_3/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_4/BiasAdd/ReadVariableOp)separable_conv2d_4/BiasAdd/ReadVariableOp2h
2separable_conv2d_4/separable_conv2d/ReadVariableOp2separable_conv2d_4/separable_conv2d/ReadVariableOp2l
4separable_conv2d_4/separable_conv2d/ReadVariableOp_14separable_conv2d_4/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_5/BiasAdd/ReadVariableOp)separable_conv2d_5/BiasAdd/ReadVariableOp2h
2separable_conv2d_5/separable_conv2d/ReadVariableOp2separable_conv2d_5/separable_conv2d/ReadVariableOp2l
4separable_conv2d_5/separable_conv2d/ReadVariableOp_14separable_conv2d_5/separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs
ш
l
@__inference_add_layer_call_and_return_conditional_losses_9446334
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:€€€€€€€€€@@А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:€€€€€€€€€@@А:€€€€€€€€€@@А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
÷
S
'__inference_add_2_layer_call_fn_9446364
inputs_0
inputs_1
identity«
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€  А*1
config_proto!

CPU

GPU2	 *0,1J 8*K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_94457262
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€  А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:€€€€€€€€€  А:€€€€€€€€€  А:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
т
l
B__inference_add_1_layer_call_and_return_conditional_losses_9445699

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:€€€€€€€€€@@А2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:€€€€€€€€€@@А2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:€€€€€€€€€@@А:€€€€€€€€€@@А:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
§
„
2__inference_separable_conv2d_layer_call_fn_9445430

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИҐStatefulPartitionedCall“
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*1
config_proto!

CPU

GPU2	 *0,1J 8*V
fQRO
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_94454212
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Њ
serving_default™
E
input_1:
serving_default_input_1:0€€€€€€€€€ААE
regression_head_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:цЊ
’©
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer-12
layer_with_weights-9
layer-13
layer-14
layer-15
layer_with_weights-10
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+З&call_and_return_all_conditional_losses
И__call__
Й_default_save_signature"Г§
_tf_keras_modelи£{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "dtype": "float32", "axis": -1}, "name": "normalization", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["normalization", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_1", "inbound_nodes": [[["separable_conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [1, 1], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["separable_conv2d_1", 0, 0, {}], ["conv2d_1", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_2", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_3", "inbound_nodes": [[["separable_conv2d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["separable_conv2d_3", 0, 0, {}], ["add", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_4", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_5", "inbound_nodes": [[["separable_conv2d_4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [3, 3], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["separable_conv2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [1, 1], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["max_pooling2d", 0, 0, {}], ["conv2d_2", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "regression_head_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "regression_head_1", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["regression_head_1", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "dtype": "float32", "axis": -1}, "name": "normalization", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["normalization", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_1", "inbound_nodes": [[["separable_conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [1, 1], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["separable_conv2d_1", 0, 0, {}], ["conv2d_1", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_2", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_3", "inbound_nodes": [[["separable_conv2d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["separable_conv2d_3", 0, 0, {}], ["add", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_4", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_5", "inbound_nodes": [[["separable_conv2d_4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [3, 3], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["separable_conv2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [1, 1], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["max_pooling2d", 0, 0, {}], ["conv2d_2", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "regression_head_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "regression_head_1", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["regression_head_1", 0, 0]]}}, "training_config": {"loss": {"regression_head_1": "mean_squared_error"}, "metrics": {"regression_head_1": ["mae", "mape"]}, "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
±"Ѓ
_tf_keras_input_layerО{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 128, 128, 3], "config": {"batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
к
state_variables
_broadcast_shape
mean
variance
	count
	variables
trainable_variables
regularization_losses
 	keras_api
+К&call_and_return_all_conditional_losses
Л__call__"Л
_tf_keras_layerс{"class_name": "Normalization", "name": "normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "normalization", "trainable": true, "dtype": "float32", "axis": -1}}
ѓ

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"И
_tf_keras_layerо{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
І
'depthwise_kernel
(pointwise_kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
+О&call_and_return_all_conditional_losses
П__call__"а	
_tf_keras_layer∆	{"class_name": "SeparableConv2D", "name": "separable_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
ђ
.depthwise_kernel
/pointwise_kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
+Р&call_and_return_all_conditional_losses
С__call__"е	
_tf_keras_layerЋ	{"class_name": "SeparableConv2D", "name": "separable_conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
с

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
+Т&call_and_return_all_conditional_losses
У__call__" 
_tf_keras_layer∞{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [1, 1], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
т
;	variables
<trainable_variables
=regularization_losses
>	keras_api
+Ф&call_and_return_all_conditional_losses
Х__call__"б
_tf_keras_layer«{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add", "trainable": true, "dtype": "float32"}}
ђ
?depthwise_kernel
@pointwise_kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
+Ц&call_and_return_all_conditional_losses
Ч__call__"е	
_tf_keras_layerЋ	{"class_name": "SeparableConv2D", "name": "separable_conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
ђ
Fdepthwise_kernel
Gpointwise_kernel
Hbias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
+Ш&call_and_return_all_conditional_losses
Щ__call__"е	
_tf_keras_layerЋ	{"class_name": "SeparableConv2D", "name": "separable_conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
ц
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
+Ъ&call_and_return_all_conditional_losses
Ы__call__"е
_tf_keras_layerЋ{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}}
ђ
Qdepthwise_kernel
Rpointwise_kernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
+Ь&call_and_return_all_conditional_losses
Э__call__"е	
_tf_keras_layerЋ	{"class_name": "SeparableConv2D", "name": "separable_conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
ђ
Xdepthwise_kernel
Ypointwise_kernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
+Ю&call_and_return_all_conditional_losses
Я__call__"е	
_tf_keras_layerЋ	{"class_name": "SeparableConv2D", "name": "separable_conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
ъ
_	variables
`trainable_variables
aregularization_losses
b	keras_api
+†&call_and_return_all_conditional_losses
°__call__"й
_tf_keras_layerѕ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [3, 3], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
т

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
+Ґ&call_and_return_all_conditional_losses
£__call__"Ћ
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [1, 1], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
ц
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
+§&call_and_return_all_conditional_losses
•__call__"е
_tf_keras_layerЋ{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}}
я
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
+¶&call_and_return_all_conditional_losses
І__call__"ќ
_tf_keras_layerі{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Й

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
+®&call_and_return_all_conditional_losses
©__call__"в
_tf_keras_layer»{"class_name": "Dense", "name": "regression_head_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "regression_head_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
Ы!m”"m‘'m’(m÷)m„.mЎ/mў0mЏ5mџ6m№?mЁ@mёAmяFmаGmбHmвQmгRmдSmеXmжYmзZmиcmйdmкqmлrmм!vн"vо'vп(vр)vс.vт/vу0vф5vх6vц?vч@vшAvщFvъGvыHvьQvэRvюSv€XvАYvБZvВcvГdvДqvЕrvЖ"
	optimizer
ю
0
1
2
!3
"4
'5
(6
)7
.8
/9
010
511
612
?13
@14
A15
F16
G17
H18
Q19
R20
S21
X22
Y23
Z24
c25
d26
q27
r28"
trackable_list_wrapper
ж
!0
"1
'2
(3
)4
.5
/6
07
58
69
?10
@11
A12
F13
G14
H15
Q16
R17
S18
X19
Y20
Z21
c22
d23
q24
r25"
trackable_list_wrapper
 "
trackable_list_wrapper
ї
wmetrics
	variables
trainable_variables
xnon_trainable_variables
ylayer_regularization_losses
regularization_losses

zlayers
И__call__
Й_default_save_signature
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
-
™serving_default"
signature_map
C
mean
variance
	count"
trackable_dict_wrapper
 "
trackable_list_wrapper
:2normalization/mean
": 2normalization/variance
: 2normalization/count
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
{metrics
	variables
trainable_variables
|non_trainable_variables
}layer_regularization_losses
regularization_losses

~layers
Л__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
':%@2conv2d/kernel
:@2conv2d/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
†
metrics
#	variables
$trainable_variables
Аnon_trainable_variables
 Бlayer_regularization_losses
%regularization_losses
Вlayers
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
;:9@2!separable_conv2d/depthwise_kernel
<::@А2!separable_conv2d/pointwise_kernel
$:"А2separable_conv2d/bias
5
'0
(1
)2"
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Гmetrics
*	variables
+trainable_variables
Дnon_trainable_variables
 Еlayer_regularization_losses
,regularization_losses
Жlayers
П__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_1/depthwise_kernel
?:=АА2#separable_conv2d_1/pointwise_kernel
&:$А2separable_conv2d_1/bias
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Зmetrics
1	variables
2trainable_variables
Иnon_trainable_variables
 Йlayer_regularization_losses
3regularization_losses
Кlayers
С__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
*:(@А2conv2d_1/kernel
:А2conv2d_1/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Лmetrics
7	variables
8trainable_variables
Мnon_trainable_variables
 Нlayer_regularization_losses
9regularization_losses
Оlayers
У__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Пmetrics
;	variables
<trainable_variables
Рnon_trainable_variables
 Сlayer_regularization_losses
=regularization_losses
Тlayers
Х__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_2/depthwise_kernel
?:=АА2#separable_conv2d_2/pointwise_kernel
&:$А2separable_conv2d_2/bias
5
?0
@1
A2"
trackable_list_wrapper
5
?0
@1
A2"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Уmetrics
B	variables
Ctrainable_variables
Фnon_trainable_variables
 Хlayer_regularization_losses
Dregularization_losses
Цlayers
Ч__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_3/depthwise_kernel
?:=АА2#separable_conv2d_3/pointwise_kernel
&:$А2separable_conv2d_3/bias
5
F0
G1
H2"
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Чmetrics
I	variables
Jtrainable_variables
Шnon_trainable_variables
 Щlayer_regularization_losses
Kregularization_losses
Ъlayers
Щ__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Ыmetrics
M	variables
Ntrainable_variables
Ьnon_trainable_variables
 Эlayer_regularization_losses
Oregularization_losses
Юlayers
Ы__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_4/depthwise_kernel
?:=АА2#separable_conv2d_4/pointwise_kernel
&:$А2separable_conv2d_4/bias
5
Q0
R1
S2"
trackable_list_wrapper
5
Q0
R1
S2"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Яmetrics
T	variables
Utrainable_variables
†non_trainable_variables
 °layer_regularization_losses
Vregularization_losses
Ґlayers
Э__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
>:<А2#separable_conv2d_5/depthwise_kernel
?:=АА2#separable_conv2d_5/pointwise_kernel
&:$А2separable_conv2d_5/bias
5
X0
Y1
Z2"
trackable_list_wrapper
5
X0
Y1
Z2"
trackable_list_wrapper
 "
trackable_list_wrapper
°
£metrics
[	variables
\trainable_variables
§non_trainable_variables
 •layer_regularization_losses
]regularization_losses
¶layers
Я__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Іmetrics
_	variables
`trainable_variables
®non_trainable_variables
 ©layer_regularization_losses
aregularization_losses
™layers
°__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
+:)АА2conv2d_2/kernel
:А2conv2d_2/bias
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Ђmetrics
e	variables
ftrainable_variables
ђnon_trainable_variables
 ≠layer_regularization_losses
gregularization_losses
Ѓlayers
£__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
ѓmetrics
i	variables
jtrainable_variables
∞non_trainable_variables
 ±layer_regularization_losses
kregularization_losses
≤layers
•__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
≥metrics
m	variables
ntrainable_variables
іnon_trainable_variables
 µlayer_regularization_losses
oregularization_losses
ґlayers
І__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
+:)	А2regression_head_1/kernel
$:"2regression_head_1/bias
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Јmetrics
s	variables
ttrainable_variables
Єnon_trainable_variables
 єlayer_regularization_losses
uregularization_losses
Їlayers
©__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
0
ї0
Љ1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
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
16"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Щ

љtotal

Њcount
њ
_fn_kwargs
ј	variables
Ѕtrainable_variables
¬regularization_losses
√	keras_api
+Ђ&call_and_return_all_conditional_losses
ђ__call__"џ
_tf_keras_layerЅ{"class_name": "MeanMetricWrapper", "name": "mae", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mae", "dtype": "float32"}}
Ы

ƒtotal

≈count
∆
_fn_kwargs
«	variables
»trainable_variables
…regularization_losses
 	keras_api
+≠&call_and_return_all_conditional_losses
Ѓ__call__"Ё
_tf_keras_layer√{"class_name": "MeanMetricWrapper", "name": "mape", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mape", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
љ0
Њ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Ћmetrics
ј	variables
Ѕtrainable_variables
ћnon_trainable_variables
 Ќlayer_regularization_losses
¬regularization_losses
ќlayers
ђ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ƒ0
≈1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
ѕmetrics
«	variables
»trainable_variables
–non_trainable_variables
 —layer_regularization_losses
…regularization_losses
“layers
Ѓ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
љ0
Њ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ƒ0
≈1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
':%@2conv2d/kernel/m
:@2conv2d/bias/m
;:9@2#separable_conv2d/depthwise_kernel/m
<::@А2#separable_conv2d/pointwise_kernel/m
$:"А2separable_conv2d/bias/m
>:<А2%separable_conv2d_1/depthwise_kernel/m
?:=АА2%separable_conv2d_1/pointwise_kernel/m
&:$А2separable_conv2d_1/bias/m
*:(@А2conv2d_1/kernel/m
:А2conv2d_1/bias/m
>:<А2%separable_conv2d_2/depthwise_kernel/m
?:=АА2%separable_conv2d_2/pointwise_kernel/m
&:$А2separable_conv2d_2/bias/m
>:<А2%separable_conv2d_3/depthwise_kernel/m
?:=АА2%separable_conv2d_3/pointwise_kernel/m
&:$А2separable_conv2d_3/bias/m
>:<А2%separable_conv2d_4/depthwise_kernel/m
?:=АА2%separable_conv2d_4/pointwise_kernel/m
&:$А2separable_conv2d_4/bias/m
>:<А2%separable_conv2d_5/depthwise_kernel/m
?:=АА2%separable_conv2d_5/pointwise_kernel/m
&:$А2separable_conv2d_5/bias/m
+:)АА2conv2d_2/kernel/m
:А2conv2d_2/bias/m
+:)	А2regression_head_1/kernel/m
$:"2regression_head_1/bias/m
':%@2conv2d/kernel/v
:@2conv2d/bias/v
;:9@2#separable_conv2d/depthwise_kernel/v
<::@А2#separable_conv2d/pointwise_kernel/v
$:"А2separable_conv2d/bias/v
>:<А2%separable_conv2d_1/depthwise_kernel/v
?:=АА2%separable_conv2d_1/pointwise_kernel/v
&:$А2separable_conv2d_1/bias/v
*:(@А2conv2d_1/kernel/v
:А2conv2d_1/bias/v
>:<А2%separable_conv2d_2/depthwise_kernel/v
?:=АА2%separable_conv2d_2/pointwise_kernel/v
&:$А2separable_conv2d_2/bias/v
>:<А2%separable_conv2d_3/depthwise_kernel/v
?:=АА2%separable_conv2d_3/pointwise_kernel/v
&:$А2separable_conv2d_3/bias/v
>:<А2%separable_conv2d_4/depthwise_kernel/v
?:=АА2%separable_conv2d_4/pointwise_kernel/v
&:$А2separable_conv2d_4/bias/v
>:<А2%separable_conv2d_5/depthwise_kernel/v
?:=АА2%separable_conv2d_5/pointwise_kernel/v
&:$А2separable_conv2d_5/bias/v
+:)АА2conv2d_2/kernel/v
:А2conv2d_2/bias/v
+:)	А2regression_head_1/kernel/v
$:"2regression_head_1/bias/v
÷2”
B__inference_model_layer_call_and_return_conditional_losses_9446240
B__inference_model_layer_call_and_return_conditional_losses_9445807
B__inference_model_layer_call_and_return_conditional_losses_9445759
B__inference_model_layer_call_and_return_conditional_losses_9446122ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
'__inference_model_layer_call_fn_9445970
'__inference_model_layer_call_fn_9445889
'__inference_model_layer_call_fn_9446273
'__inference_model_layer_call_fn_9446306ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
"__inference__wrapped_model_9445383ј
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *0Ґ-
+К(
input_1€€€€€€€€€АА
ф2с
J__inference_normalization_layer_call_and_return_conditional_losses_9446321Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ў2÷
/__inference_normalization_layer_call_fn_9446328Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ґ2Я
C__inference_conv2d_layer_call_and_return_conditional_losses_9445396„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
З2Д
(__inference_conv2d_layer_call_fn_9445404„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
ђ2©
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_9445421„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
С2О
2__inference_separable_conv2d_layer_call_fn_9445430„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
ѓ2ђ
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_9445447Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ф2С
4__inference_separable_conv2d_1_layer_call_fn_9445456Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
§2°
E__inference_conv2d_1_layer_call_and_return_conditional_losses_9445468„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Й2Ж
*__inference_conv2d_1_layer_call_fn_9445476„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
к2з
@__inference_add_layer_call_and_return_conditional_losses_9446334Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_add_layer_call_fn_9446340Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѓ2ђ
O__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_9445493Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ф2С
4__inference_separable_conv2d_2_layer_call_fn_9445502Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
ѓ2ђ
O__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_9445519Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ф2С
4__inference_separable_conv2d_3_layer_call_fn_9445528Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
м2й
B__inference_add_1_layer_call_and_return_conditional_losses_9446346Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_add_1_layer_call_fn_9446352Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѓ2ђ
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_9445545Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ф2С
4__inference_separable_conv2d_4_layer_call_fn_9445554Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
ѓ2ђ
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_9445571Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ф2С
4__inference_separable_conv2d_5_layer_call_fn_9445580Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
≤2ѓ
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_9445586а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ч2Ф
/__inference_max_pooling2d_layer_call_fn_9445592а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
•2Ґ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9445604Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
К2З
*__inference_conv2d_2_layer_call_fn_9445612Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
м2й
B__inference_add_2_layer_call_and_return_conditional_losses_9446358Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_add_2_layer_call_fn_9446364Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
љ2Ї
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_9445619а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ґ2Я
:__inference_global_average_pooling2d_layer_call_fn_9445625а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ш2х
N__inference_regression_head_1_layer_call_and_return_conditional_losses_9446374Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ё2Џ
3__inference_regression_head_1_layer_call_fn_9446381Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
4B2
%__inference_signature_wrapper_9446004input_1
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 »
"__inference__wrapped_model_9445383°!"'()./056?@AFGHQRSXYZcdqr:Ґ7
0Ґ-
+К(
input_1€€€€€€€€€АА
™ "E™B
@
regression_head_1+К(
regression_head_1€€€€€€€€€е
B__inference_add_1_layer_call_and_return_conditional_losses_9446346ЮlҐi
bҐ_
]ЪZ
+К(
inputs/0€€€€€€€€€@@А
+К(
inputs/1€€€€€€€€€@@А
™ ".Ґ+
$К!
0€€€€€€€€€@@А
Ъ љ
'__inference_add_1_layer_call_fn_9446352СlҐi
bҐ_
]ЪZ
+К(
inputs/0€€€€€€€€€@@А
+К(
inputs/1€€€€€€€€€@@А
™ "!К€€€€€€€€€@@Ае
B__inference_add_2_layer_call_and_return_conditional_losses_9446358ЮlҐi
bҐ_
]ЪZ
+К(
inputs/0€€€€€€€€€  А
+К(
inputs/1€€€€€€€€€  А
™ ".Ґ+
$К!
0€€€€€€€€€  А
Ъ љ
'__inference_add_2_layer_call_fn_9446364СlҐi
bҐ_
]ЪZ
+К(
inputs/0€€€€€€€€€  А
+К(
inputs/1€€€€€€€€€  А
™ "!К€€€€€€€€€  Аг
@__inference_add_layer_call_and_return_conditional_losses_9446334ЮlҐi
bҐ_
]ЪZ
+К(
inputs/0€€€€€€€€€@@А
+К(
inputs/1€€€€€€€€€@@А
™ ".Ґ+
$К!
0€€€€€€€€€@@А
Ъ ї
%__inference_add_layer_call_fn_9446340СlҐi
bҐ_
]ЪZ
+К(
inputs/0€€€€€€€€€@@А
+К(
inputs/1€€€€€€€€€@@А
™ "!К€€€€€€€€€@@Аџ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_9445468С56IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ≥
*__inference_conv2d_1_layer_call_fn_9445476Д56IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А№
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9445604ТcdJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ і
*__inference_conv2d_2_layer_call_fn_9445612ЕcdJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЎ
C__inference_conv2d_layer_call_and_return_conditional_losses_9445396Р!"IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ∞
(__inference_conv2d_layer_call_fn_9445404Г!"IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ё
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_9445619ДRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ µ
:__inference_global_average_pooling2d_layer_call_fn_9445625wRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "!К€€€€€€€€€€€€€€€€€€н
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_9445586ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≈
/__inference_max_pooling2d_layer_call_fn_9445592СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€–
B__inference_model_layer_call_and_return_conditional_losses_9445759Й!"'()./056?@AFGHQRSXYZcdqrBҐ?
8Ґ5
+К(
input_1€€€€€€€€€АА
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ –
B__inference_model_layer_call_and_return_conditional_losses_9445807Й!"'()./056?@AFGHQRSXYZcdqrBҐ?
8Ґ5
+К(
input_1€€€€€€€€€АА
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ѕ
B__inference_model_layer_call_and_return_conditional_losses_9446122И!"'()./056?@AFGHQRSXYZcdqrAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ѕ
B__inference_model_layer_call_and_return_conditional_losses_9446240И!"'()./056?@AFGHQRSXYZcdqrAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ І
'__inference_model_layer_call_fn_9445889|!"'()./056?@AFGHQRSXYZcdqrBҐ?
8Ґ5
+К(
input_1€€€€€€€€€АА
p

 
™ "К€€€€€€€€€І
'__inference_model_layer_call_fn_9445970|!"'()./056?@AFGHQRSXYZcdqrBҐ?
8Ґ5
+К(
input_1€€€€€€€€€АА
p 

 
™ "К€€€€€€€€€¶
'__inference_model_layer_call_fn_9446273{!"'()./056?@AFGHQRSXYZcdqrAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ "К€€€€€€€€€¶
'__inference_model_layer_call_fn_9446306{!"'()./056?@AFGHQRSXYZcdqrAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ "К€€€€€€€€€Њ
J__inference_normalization_layer_call_and_return_conditional_losses_9446321p9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ Ц
/__inference_normalization_layer_call_fn_9446328c9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ ""К€€€€€€€€€ААѓ
N__inference_regression_head_1_layer_call_and_return_conditional_losses_9446374]qr0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€
Ъ З
3__inference_regression_head_1_layer_call_fn_9446381Pqr0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€з
O__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_9445447У./0JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ њ
4__inference_separable_conv2d_1_layer_call_fn_9445456Ж./0JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аз
O__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_9445493У?@AJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ њ
4__inference_separable_conv2d_2_layer_call_fn_9445502Ж?@AJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аз
O__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_9445519УFGHJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ њ
4__inference_separable_conv2d_3_layer_call_fn_9445528ЖFGHJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аз
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_9445545УQRSJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ њ
4__inference_separable_conv2d_4_layer_call_fn_9445554ЖQRSJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аз
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_9445571УXYZJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ њ
4__inference_separable_conv2d_5_layer_call_fn_9445580ЖXYZJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Ад
M__inference_separable_conv2d_layer_call_and_return_conditional_losses_9445421Т'()IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Љ
2__inference_separable_conv2d_layer_call_fn_9445430Е'()IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А÷
%__inference_signature_wrapper_9446004ђ!"'()./056?@AFGHQRSXYZcdqrEҐB
Ґ 
;™8
6
input_1+К(
input_1€€€€€€€€€АА"E™B
@
regression_head_1+К(
regression_head_1€€€€€€€€€