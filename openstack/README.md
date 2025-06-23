# One-time setup of a docker container for openstack and create a security group
```bash
docker run -it --name create-cloud ubuntu:latest
docker cp clouds.yaml /root
docker cp ~/.ssh/create_cloud:/root/.ssh/
docker start -ai create-cloud

DEBIAN_FRONTEND=noninteractive
mkdir /root/.ssh
apt-get update
apt-get install -y python3-pip python3-venv curl ca-certificates jq ssh vim
cd
curl -OJ https://docs.er.kcl.ac.uk/resources/KCL-ER-Root-CA.crt
cp KCL-ER-Root-CA.crt /usr/local/share/ca-certificates
update-ca-certificates
openssl x509 -in KCL-ER-Root-CA.crt -text >> .venv/lib/python3.12/site-packages/certifi/cacert.pem
pip install python-openstackclient
python3 -m venv .venv
security group create allow_all
openstack security group rule create \
  --protocol any \
  --ingress \
  --ethertype IPv4 \
  --remote-ip 0.0.0.0/0 \
  allow_all
openstack security group rule create \
  --protocol any \
  --ingress \
  --ethertype IPv6 \
  --remote-ip ::/0 \
  allow_all

openstack security group create allow_ssh
openstack security group rule create \
  --ingress --protocol tcp --dst-port 22 \
  --remote-ip 0.0.0.0/0 \
  allow_ssh
```

# Evrytime usage
```bash
docker start -ai create-cloud
source .venv/bin/activate
```

## Queries
```bash
openstack --os-cloud=openstack server list
openstack --os-cloud=openstack flavor list
openstack --os-cloud=openstack flavor show 1cpu1ram
openstack --os-cloud=openstack image list
openstack --os-cloud=openstack network list
openstack --os-cloud=openstack key-name list
openstack --os-cloud=openstack keypair list
openstack --os-cloud=openstack security group list
```

## Create a VM
```bash
# jammy
openstack --os-cloud=openstack server create   --flavor 1cpu1ram   --image public_ubuntu-jammy_adjoin_2024-07-08_125408   --network external_4003   --key-name create-cloud  my-server

#focal
openstack --os-cloud=openstack server create   --flavor 1cpu1ram/4cpu8ram   --image "KCL Ubuntu 20.04"   --network external_4003   --key-name create-cloud  my-server

# then run this (for both)
openstack --os-cloud=openstack server add security group my-server allow_ssh
openstack --os-cloud=openstack server show my-server -f value -c addresses
```

## Connect to a VM
```bash
vm_ip=$(openstack server show my-server -f json -c addresses | jq -r ".addresses.external_4003.[0]")
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/create_cloud ubuntu@$vm_ip
```

## Delete a VM
```bash
openstack --os-cloud=openstack server delete my-server
```

# Setup the VM
```bash
sudo apt-get update
sudo apt-get install -y cifs-utils unzip
cat <<EOF | sudo tee /etc/smb_creds > /dev/null
username=er_smb_group3819
password=[REDACTED]
domain=kclad
EOF
sudo chmod 600 /etc/smb_creds
#sudo mkdir -p /rds/dh_golden_triangle
#echo "//rds.er.kcl.ac.uk/dh_golden_triangle /rds/dh_golden_triangle cifs credentials=/etc/smb_creds 0 0" | sudo tee -a /etc/fstab > /dev/null
sudo mkdir -p /hpc/scratch/prj/dh_golden_triangle
echo "//smb.create.kcl.ac.uk/hpc/scratch/prj/dh_golden_triangle /hpc/scratch/prj/dh_golden_triangle cifs credentials=/etc/smb_creds 0 0" | sudo tee -a /etc/fstab > /dev/null
sudo mount /hpc/scratch/prj/dh_golden_triangle
```