#!/bin/sh

IP_ADDR=192.168.32.30

curl -s -o /dev/null -w "Shopping (7770): %{http_code}\n" http://$IP_ADDR:7770
curl -s -o /dev/null -w "Shopping Admin (7780): %{http_code}\n" http://$IP_ADDR:7780
curl -s -o /dev/null -w "Forum (9999): %{http_code}\n" http://$IP_ADDR:9999
curl -s -o /dev/null -w "Wikipedia (8888): %{http_code}\n" http://$IP_ADDR:8888
curl -s -o /dev/null -w "Map (3000): %{http_code}\n" http://$IP_ADDR:3000
curl -s -o /dev/null -w "GitLab (8023): %{http_code}\n" http://$IP_ADDR:8023
curl -s -o /dev/null -w "Map tile: %{http_code}\n" http://$IP_ADDR:3000/tile/0/0/0.png
curl -s -o /dev/null -w "Homepage: %{http_code}\n" http://$IP_ADDR:4399

export SHOPPING="http://$IP_ADDR:7770"
export SHOPPING_ADMIN="http://$IP_ADDR:7780/admin"
export REDDIT="http://$IP_ADDR:9999"
export GITLAB="http://$IP_ADDR:8023"
export MAP="http://$IP_ADDR:3000"
export WIKIPEDIA="http://$IP_ADDR:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://$IP_ADDR:4399"
