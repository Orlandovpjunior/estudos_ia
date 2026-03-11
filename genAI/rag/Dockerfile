# Use the AWS base image for python 3.12
FROM public.ecr.aws/lambda/python:3.12

# install build-essential compiler and tools
RUN microdnf update -y && microdnf install -y gcc-c++ make

# Copy the requirements.txt file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# install the libs in requirements.txt file
RUN pip install -r requirements.txt

# COPY PDF file
COPY data/visao-estereo-rev.pdf ${LAMBDA_TASK_ROOT}/visao-estereo-rev.pdf

# COPY the fuction code
COPY simpleRag.py ${LAMBDA_TASK_ROOT}

# Set permission to make the file executable
RUN chmod +x simpleRag.py

CMD [ "simpleRag.lambda_handler" ]