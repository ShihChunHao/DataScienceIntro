# Information Security Homework1

###### 109511119 施竣皓 2022/10/30

<!--*Zoom 於 2020 年時被發現使用 AES-128-ECB 使得通訊的安全性低落*-->
#### *Zoom's use of AES-128-ECB has been found to compromise the security of communications by 2020*

## Introduction
<!--Zoom是一個提供視訊會議服務的公司，雖然他已經在美國NASDAQ上市，但其實他是一間位於中國的企業。

2020年，有人發現Zoom會把非中國的用戶資料傳到設立在中國的伺服器，這不禁讓人們擔心資訊安全——由於中國的政府可能會要求Zoom提供影像內容。（事實上，中國政府也曾透過Zoom視訊會議，抓到透過視訊會議集會的中國基督教徒。）

從上述的例子中我們不難看出Zoom在過往是一間對資訊安全保護不佳的公司。而今天我們將探討的是，Zoom被發現使用AES-128-ECB方法加密的緣由，以及後面他推出更新版的一些做法。-->
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Zoom is a company that provides video conferencing services, and although it is listed on the NASDAQ in the US, it is actually a China-based company.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;In 2020, it was discovered that Zoom was transferring non-Chinese user data to servers in China, raising concerns about information security - as the Chinese government might ask Zoom to provide video content. (In fact, the Chinese government has caught Chinese Christians meeting via video conferencing via Zoom.)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;It is easy to see from the above examples that Zoom has been a poorly secured company in the past. Today we will look at the reasons why Zoom was found to be using the AES-128-ECB method of encryption, and some of the ways in which it has since launched an updated version.




## Found to be encrypted using AES-128-ECB 

<!--## 如何被發現是採用 AES-128-ECB-->
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;According to a report released by the Canadian group Citizen's Lab, the group monitors network packets primarily through the Wireshark software. (Note: Wireshark is a network protocol analyzer that allows users to analyse network protocols.)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Citizen's Lab conducted a traffic analysis of the network during the Zoom conference and the results showed that most of the traffic was transmitted over UDP (User Packet Protocol), the transport layer of the network protocol, which is a method of separating data into packets for transmission and is often used for services that require immediate presentation. 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;UDP does not require a unique identifier for transmission; in other words, the sender sends the data and does not wait for an acknowledgement signal from the receiver (no three-way handshake), but keeps sending the packet for testing. The group also found that at the application level Zoom uses the RTP (Real-time Transport Protocol) protocol for the transmission of audio and video.

<!--根據加拿大團體“Citizen's Lab”釋出的報告，該團體主要透過Wireshark軟體進監管網路的封包。(註：Wireshark是一個網路協議的分析儀，使用者可以透過這個軟體進行網路協議的分析。)

Citizen's Lab 在Zoom的會議期間，進行了網路的流量分析，結果顯示大多數流量是經過UDP傳輸。UDP（用戶資料包協定）屬於網路協定中的傳輸層，該協定是將資料分隔成封包再傳輸的一種方法，通常用於需要即時呈現的服務。此種方法不需要唯一識別碼就能進行傳輸；換言之發送端送出資料後，不會等待接收端的確認訊號（不進行三向交握），會一直發送封包測試。而同時，該團體也發現在應用層Zoom使用了RTP(Real-time Transport Protocol)協定來傳輸音訊和影片。-->

### The process of decrypting encrypted films
<!--###破解加密影片的過程-->
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Citizen's Lab found that on packets with RTP load starters 0x05100100 at UDP, the multimedia type (PT) transmitted is mostly 0b1100010, which according to the definition of PT in GB28181, 0b1100010 (98) represents a video with H.264 encoding method.
<center>


**Table1. RTSP Protocol Series PT**  [1]


| **Type** | **Encode Name** | **Clock Rate** | **Channel** | **Media** | 
|----------|-----------------|----------------|-------------|-----------|
| 4        | G.723           | 8k Hz          | 1           | Audio     |               |      |               |      |               |
| 8        | PCMA(G.711 A)   | 8k Hz          | 1           | Audio     |               |      |               |      |               |
| 9        | G.722           | 8k Hz          | 1           | Audio     |               |      |               |      |               |
| 18       | G.729           | 8k Hz          | 1           | Audio     |               |      |               |      |               |
| 20       | SVACA           | 8k Hz          | 1           | Audio     |               |      |               |      |               |
| 96       | PS              | 90k Hz         |             | Video     |               |      |               |      |               |
| 97       | MPEG-4          |                |             | Video     |               |      |               |      |               |
| 98       | H.264           |                |             |           |               |      |               |      |               |
| 99       | SAVC            |                |             |           |               |      |               |      |               |
</center>


&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Videos encoded according to the H.264 encoding method are assembled via the Network Abstraction Layer Unit (NALU) for data storage and transmission. However, the Citizen's Lab found that the nal_unit_type was set to zero after the NALU was reassembled, indicating UNKNOWN, so the group became suspicious of the format that Zoom had created.

<div STYLE="page-break-after: always;"></div>

<center>
![alt](https://ask.qcloudimg.com/http-save/yehe-5796036/n525welvaa.png?imageView2/2/w/1620 'Title')

**Fig1 Introduction to NALU.** [2]</center>

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;A check of Zoom's memory during the conference revealed that the AES-128 key in the memory was associated with the string conf.skey, which the group literally considered to be the conference key.


<center>**Table2. NALU types**  [2]</center>

<center>

| **NALU_type** | **NALU_INFO** |
|-----------|-----------------|
| 0         | Unknown         |
| 1         | SLCIE           |
| 2         | SLICE_DPA       |
| 3         | SLICE_DPB       |
| 4         | SLICE_DPC       |
| 5         | SLICE_IDR       |
| 6         | SEI             |
| 7         | SPS             |
| 8         | PPS             |
| 9         | AUD             |
| 10        | Series END      |
| 11        | Code Stream END |
</center>





&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Finally, Citizen's Lab decrypted the AES-128 key, wrote the data to the disk and successfully played the video file using the VLC command.

<!--Citizen's Lab在UDP發現了RTP負載開頭為0x05100100的封包上，其傳輸的多媒體型別（PT）大多為0b1100010，根據GB28181中對PT的定義，0b1100010（98）代表的是H.264編碼方法的影片。


table10 RTSP協議系列 PT https://www.gushiciku.cn/pl/ptRK/zh-tw
根據H.264編碼方法的影片會通過Network Abstraction Layer Unit (NALU) 組成，以便進行數據的存儲與傳輸。然而Citizen's Lab在重組NALU後發現其型別值（nal_unit_type）皆被設置為零，表示為UNKNOWN，因此該團體便開始懷疑Zoom所訂製的格式。

fig1 NALU介紹 https://cloud.tencent.com/developer/article/1746993
table1 NALU類型 https://cloud.tencent.com/developer/article/1746993
在會議期間對Zoom記憶體的檢查結果顯示，內存中的AES-128密鑰與字串conf.skey相關，從字面上來說，該團體認為他是會議室金鑰（Conference Key）。

最後，Citizen's Lab透過AES-128金鑰進行解密，將資料寫入磁碟，並用VLC指令成功播放影片檔案。-->

[3] 
```bash=
$ vlc raw.h264 --demux h264
```



### The process of decrypting encrypted audio

<!--###破解加密音訊的過程-->
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;At the same time, Citizen's Lab found that packets with a load header of 0x050f0100 transmitted a multimedia type (PT) of mostly 0b1110000 (112); and they found that the timestamp increased with packet size640.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;In the paper <Fast RTP Detection and Codecs Classifification in Internet Traffiffic>, a method for determining the type of audio decoder from examining RTP packets is mentioned - Skype developed the The SILK encoder developed by Skype has a sample rate of 640, so the group guessed that Zoom was using a SILK encoder.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Finally, Citizen's Lab decrypted the AES-128 key and obtained a SILK codec, which verified the previous guess, and used the codec to obtain an audio mp3 file of one of the participants.

<!--同時，Citizen's Lab發現了負載開頭為0x050f0100的封包上，其傳輸的多媒體型別（PT）大多為0b1110000（112）；並且他們發現時間戳隨著封包增加而增加640。

在<Fast RTP Detection and Codecs Classifification in Internet Traffiffic>這篇論文中，提到了從檢視RTP封包來判斷音訊解碼器型別的方法——Skype開發的SILK編碼器取樣率為640，因此該團體猜測Zoom使用了SILK編碼器。

最後，Citizen's Lab透過AES-128金鑰進行解密，得到了一個SILK的編碼器，驗證了之前的猜想，並透過該編碼器得到一個參與者的音訊mp3檔案。-->


<center>[3]  ```bash=
$ sh converter.sh raw.silk mp3
```</center>

<div STYLE="page-break-after: always;"></div>

## What is GCM
<!--在被發現使用AES-128-ECB加密後，Zoom隨即更新的軟體的加密，採用了AES-256-GCM。那什麼是GCM呢？

GCM則是基於CTR方法做延伸，一樣使用到了Initial Vector(IV)和「後一組的密碼生成需要前一組的密文參與」的概念。同時，他是一個Stream Cipher，因此執行速度比Block Cipher還要快，可以進行併行運算，大大加快加密解密的時間。此外，GCM也提供了可驗證性的方法，信息的接收者需要確認到身份的驗證。-->
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;After being discovered to be using AES-128-ECB encryption, Zoom has updated its software to use AES-256-GCM.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;GCM is an extension of the CTR method, which also uses Initial Vector(IV) and the concept that "the generation of a subsequent cipher requires the participation of the previous cipher". It is also a Stream Cipher, so it is faster than a Block Cipher and can perform concurrent operations, which greatly speeds up the encryption and decryption time. In addition, GCM also provides a method of verifiability where the recipient of the message needs to confirm to the identity verification.


<center> ![alt](https://i.imgur.com/SeSEAq8.png 'Title') </center>
<center>  **Fig2 GCM flowchart**  [4] </center>

## Why AES-256-GCM is more secure than AES-128-ECB
<!--## 為什麼 AES-256-GCM 比 AES-128-ECB 安全-->
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Compared to AES-256-GCM and AES-128-ECB, apart from the increase in the number of key bits (from 128 bits to 256 bits), the most important difference is the difference between GCM and ECB.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Traditionally, ECB uses the same key for encryption. The principle is simply to cut the plaintext to size and encrypt the cut plaintext with the same key. This is a very dangerous way of encrypting images, and as long as the image does not change much, the outline of the encrypted file will still be visible. (For example, the background will be encrypted in the same colour, so that the outline of the object can be easily seen)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;From the above brief analysis, we can conclude that AES-256-GCM is more secure than AES-128-ECB for the following reasons: the difference in the number of bits encrypted and the problem of the ECB encrypted image.

<!--AES-256-GCM 和 AES-128-ECB相比之下，除了金鑰位元數的提升（從128 bits提升到256 bits），最重要的是GCM和ECB的不同之處。

傳統上的使用的ECB，是使用相同的Key進行加密，其原理只是將明文按照大小進行切割，再用同樣的金鑰加密這些切割好的明文。這種方式的加密對於影像的加密非常危險，只要影像變化不大，加密過後的檔案基本上還是可以看出他的輪廓。（舉個例子，背景會被加密為同一顏色，就能輕易地看出物體的輪廓）

fig3 (a) original (b) ECB (c) 有IV原理的加密模式

通過以上的簡單分析，我們可以得出AES-256-GCM 比 AES-128-ECB 安全的理由：加密位元數的差異 以及 ECB加密影像的問題。-->


***
<div STYLE="page-break-after: always;"></div>

#### *Zerologon story*

## Introduction
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;The Zerologon vulnerability is categorised in the Common Vulnerability Scoring System (CVSS) as a vulnerability with a severity score of 10 out of 10.
<!--Zerologon 漏洞在 Common Vulnerability Scoring System (CVSS) 的評分被歸類為嚴重性達到滿分 10 分的漏洞。-->

## What is Zerologon
<!--Zerologon 是在加密時候將初始化向量 (Initialization Vector，簡稱 IV) 都設為零，所引發的安全問題。由於將IV都設為零，就導致了被駭客入侵的可能性增加。
根據一位在 Secura 工作的荷蘭研究員 Tom Tervoort 所述，雖然在一般網站上登入的次數是有限制的，但是電腦開機的登入介面卻沒有這樣的限制。當駭客發現一個可以使密文全零的金鑰，就能夠去停用加密的憑證，導致接下來的傳輸都能以明文的方式傳送。-->

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Zerologon is a security problem caused by setting the Initialization Vector (IV) to zero during encryption. By setting all IVs to zero, the possibility of hacking is increased.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;According to Tom Tervoort, a Dutch researcher working at Secura, while there is a limit to the number of logins on a normal website, there is no such limit on the computer's boot interface. When a hacker finds a key that can make the ciphertext all-zero, it can deactivate the encrypted certificate, allowing subsequent transmissions to be made in plaintext.

<!--## 何謂 AES-CFB8？ AES-CFB8 與課堂上的 AES-CFB 有何差別？-->
## What is AES-CFB8? What is the difference between AES-CFB8 and AES-CFB?
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;AES-CFB8 is an extension of AES-CFB.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;The core idea is to first encrypt the cipher text with a cryptographic algorithm and then XOR it with the plain text blocks.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;On aria-128-cfb, each iteration encrypts 128 bits and produces 128 bits of ciphertext. AES-CFB8, on the other hand, encrypts 8 bits per iteration and generates 8 bits of ciphertext at the same time.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Although at first glance this method is inefficient, as AES-CFB8 requires 16 times as many blocks to operate; however, this splitting method will only result in a single encryption error when the content is corrupted, and thus makes it easier to check for errors.

<!--AES-CFB8 是AES-CFB的一種延伸。

課堂上的 AES-CFB 是一種Block Cipher的加密方法，其核心想法是先把密文透過加密演算法加密，再與明文區塊做XOR運算。

在aria-128-cfb上，每一次的迭代會加密128個bits，同時產生128bits的密文。 AES-CFB8則代表每一次的迭代會加密8個bits，同時產生8bits的密文。

雖然乍看之下這樣的方法效率很低，AES-CFB8需要使用16倍的block來操作；然而當內容產生損壞，這樣分割的方法，只會導致一次的加密產生錯誤，因此反而能方便檢查有沒有出現錯誤。-->
***
<div STYLE="page-break-after: always;"></div>
## What I think
<!--這次的作業我覺得最有趣的地方在於理解Citizen's Lab破解的每個過程。為了理解Citizen's Lab報告所說明的一些專業術語，我也去看了很多型態識別碼的文件，也實際看了國外專家透過wireshark抓取封包來分析的一些影片。

我覺得這些東西的編碼與MIPS的編碼有異曲同工之妙，都在於制定好標準後，讓行業人員根據這樣的規範來進行開發。

最後，在上課之後理解了各種專業名詞後，也透過這次的作業開始慢慢看得懂一些跟資訊安全相關的新聞。很期待接下來的課程！
-->
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;The most interesting part of this assignment for me was to understand the process of the Citizen's Lab crack. In order to understand some of the terminology explained in the Citizen's Lab report, I also went through a lot of documentation on pattern recognition codes and actually watched some videos of foreign experts capturing packets through wireshark to analyse them.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;I think the coding of these things is similar to the MIPS coding, in that they are all based on a standard that is developed for industry personnel to follow.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Finally, after understanding the terminology, I started to understand some of the news related to information security through this assignment. 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;I am looking forward to the upcoming courses!

## Reference
[1] Reference: GB28181 Document : https://www.gushiciku.cn/pl/ptRK/zh-tw

[2] 音视频压缩：H264码流层次结构和NALU详解：https://ask.qcloudimg.com/http-save/yehe-5796036/n525welvaa.png?imageView2/2/w/1620

[3] Reference:Move Fast and Roll Your Own Crypto A Quick Look at the Confidentiality of Zoom Meetings:https://citizenlab.ca/2020/04/move-fast-roll-your-own-crypto-a-quick-look-at-the-confidentiality-of-zoom-meetings/

[4]  加密演算法要注意的加密模式 : https://ithelp.ithome.com.tw/articles/10249953

