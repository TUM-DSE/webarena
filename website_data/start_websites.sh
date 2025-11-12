#!/bin/sh

IP_ADDR=192.168.32.30

echo "Starting shopping website..."
docker load --input shopping_final_0712.tar
docker run --name shopping -p 7770:80 -d shopping_final_0712
echo "Shopping website active "

echo "Starting CMS website..."
docker load --input shopping_admin_final_0719.tar
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719

echo "Starting Reddit..."
docker load --input postmill-populated-exposed-withimg.tar
docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg

echo "Starting Gitlab..."
docker load --input gitlab-populated-final-port8023.tar
docker run --name gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start

echo "Starting Wikipedia..."
docker run -d --name=wikipedia --volume=./:/data -p 8888:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim

echo "Waiting ~5 mins for all services to start..."
sleep 320
echo "Configuring websites..."
docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$IP_ADDR:7770" # no trailing slash
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  "UPDATE core_config_data SET value=\"http://$IP_ADDR:7770/\" WHERE path = \"web/secure/base_url\";"
docker exec shopping /var/www/magento2/bin/magento cache:flush

docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$IP_ADDR:7780" # no trailing slash
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  "UPDATE core_config_data SET value=\"http://$IP_ADDR:7780/\" WHERE path = \"web/secure/base_url\";"
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush

docker exec gitlab sed -i "s|^external_url.*|external_url 'http://$IP_ADDR:8023'|" /etc/gitlab/gitlab.rb
docker exec gitlab gitlab-ctl reconfigure

echo "Testing connectivity..."

curl -s -o /dev/null -w "Shopping (7770): %{http_code}\n" http://$IP_ADDR:7770
curl -s -o /dev/null -w "Shopping Admin (7780): %{http_code}\n" http://$IP_ADDR:7780
curl -s -o /dev/null -w "Forum (9999): %{http_code}\n" http://$IP_ADDR:9999
curl -s -o /dev/null -w "Wikipedia (8888): %{http_code}\n" http://$IP_ADDR:8888
curl -s -o /dev/null -w "Map (3000): %{http_code}\n" http://$IP_ADDR:3000
curl -s -o /dev/null -w "GitLab (8023): %{http_code}\n" http://$IP_ADDR:8023
curl -s -o /dev/null -w "Map tile: %{http_code}\n" http://$IP_ADDR:3000/tile/0/0/0.png
