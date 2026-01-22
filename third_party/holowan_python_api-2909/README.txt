【使用方法】

1. Windows系统运行install.bat，Mac系统在终端执行sudo python ./setup.py install，即可安装HoloWAN Python API库
2. 在Python脚本代码中通过“from holowan.HoloWAN import HoloWAN”导入holowan
3. 具体调用方法详见API说明文档

【更新日志】

2.4.3版本:
	1. 新增马尔可夫丢包模型接口
	2. 新增Recorder相关接口
2.4.4版本:
	1. 删除了无用的第三方包
2.5.0版本:
	1. 删除recorder相关接口
	2. 新增playback相关接口
2.6.0版本:
	1. 根据HoloWAN新的xml结构，修改了带宽设置函数set_path_Bandwidth_Fixed()
2.7.0版本:
	1. 调整了以太网间隙占用Frame Overhead的取值范围，现在最小值可取到0
	2. 新增偏好设置相关接口
		是否清空buffer
		是否开启巨型帧
		设置报文时延起点
		设置带宽计算逻辑
		设置语言
2.8.0版本:
	1. 根据holowan服务端修改的链路配置xml格式，修改了path_config.xml文件内有关链路带宽设置的xml格式

2.9.0版本:
	1. 新增线路选择相关接口

1980版本：
	1. 更新Playback相关接口
	2. 新增BER - Range功能接口
	3. 更改版本命名方式，重新编写API说明文档

2180版本：
	1. 根据HoloWAN的接口更改，修改函数get_path_graph_current_dataGroup()的参数
	2. 修改Corruption功能的xml配置参数的定义，<s>标签为1时表示使用BER Normal，为2时表示使用BER Range
	3. 修正API文档中的一些变量类型错误
	4. 修复Modify功能不能设置为off的BUG（对应的函数为close_message_Modify()）
	5. 新增Delay-Jitter功能的API
	6. 根据holowan后端的修改，更新reset_engine、reset_path接口的请求地址，并添加add_path_and_enable接口
	7. 添加对Red队列丢弃算法的支持

2900版本：
    1. 添加v2版本python api，完全兼容旧版本api

2901版本：
    1. 添加报文分类规则时的端口号和前端对应。
    2. 修改带宽限制令牌桶参数错误。
    3. 添加重新排列报文分类器规则的方法。
    4. 添加修改指定索引位置报文分类器规则的方法。
    5. 添加在指定位置插入报文分类器规则的方法。
    6. 添加可用于配置GPTs的文本文件API GEMS

2902版本：
    1. 增加报文分类规则按给定的索引位置删除的方法。
    2. 更正sequential容器的一些错误。


2903版本：
    1. 修改 v1 api 的modify：添加matchSwitch和delete接口
    2. v2 api 新增损伤的enable，version

2904版本：
    1. 修复 v2 reset path错误开启问题
    2. v1,v2 set_playback_status 分开两个接口
    3. v1,v2 获取、设置时间同步
    4. v2 queue limit RED 解析失败
    5. v2 ipv6报文分类规则参数检查错误
    6. v2 queue limit 删除无用的 qdm 字段

2905版本：
    1. v2: 处理报文分类规则的 label 问题；部分规则 API 生成默认的自定义名称；自定义规则名称的 set_custom_name
    2. v2: 更正 token bucket 的一些xml字段错误
    3. v2: path.reset 在gui 2.0时损伤应该是禁用状态；一些有disable的损伤在gui2.0的问题

2906版本：
    1. 打包 TypeFilter 、PathFilter、 DirectionFilter，与其他过滤方式保持一致，方便用户使用。
    2. 补充 reset filtersetting  的接口，用于重置 pIxel 过滤器
    3. 修复 pixel typefilter中 all 和其他 damage 的关系问题

2907版本：
    1. 新增获取 pixel 抓包数据包总数
    2. 新增获取 pixel 抓包数据包信息

2908版本：
    1. 增加 BandwidthBidirectional
    2. 增加 LossCount
    3. 增加 ModifyRanges,ModifyInsert,ModifyDelete,ModifyExchange,ModifyCount
    4. 增加 BackgroundUtilizationPCAP
    5. 增加 DelayCustomizedFile
    6. 增加 BERCount

2909版本：
    1.处理 HoloWAN 使用 IPv6 管理地址
    2.处理 HoloWAN 登录验证
