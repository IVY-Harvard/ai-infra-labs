"""
设备发现模块测试

测试RDMA设备扫描器和拓扑映射器，使用模拟的subprocess输出。
"""

import unittest
from unittest.mock import patch, MagicMock
import json

from src.discovery.rdma_device_scanner import (
    RDMADeviceScanner, RDMADevice, RDMAPort, NodeRDMAInfo,
    PortState, LinkLayer
)
from src.discovery.topology_mapper import (
    TopologyMapper, NetworkTopology, NetworkNode, NetworkLink,
    NodeType, TopologyType, SwitchTier
)


class TestRDMADeviceScanner(unittest.TestCase):
    """RDMA设备扫描器测试"""

    def setUp(self):
        """测试初始化"""
        self.config = {
            "nodes": [
                {"hostname": "node01", "ip": "192.168.1.1"},
                {"hostname": "node02", "ip": "192.168.1.2"},
            ],
            "ssh_user": "root",
            "ssh_key": "~/.ssh/id_rsa",
            "ssh_timeout": 10,
            "max_scan_workers": 2,
        }
        self.scanner = RDMADeviceScanner(self.config)

    def test_parse_ibv_devices(self):
        """测试解析ibv_devices输出"""
        output = """    device                 node GUID
    ------              ----------------
    mlx5_0              7cfe900300e1a820
    mlx5_1              7cfe900300e1a821
"""
        devices = self.scanner._parse_ibv_devices(output)
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0], "mlx5_0")
        self.assertEqual(devices[1], "mlx5_1")

    def test_parse_ibv_devices_empty(self):
        """测试解析空的ibv_devices输出"""
        output = """    device                 node GUID
    ------              ----------------
"""
        devices = self.scanner._parse_ibv_devices(output)
        self.assertEqual(len(devices), 0)

    def test_parse_ibstat_device(self):
        """测试解析ibstat输出"""
        output = """CA 'mlx5_0'
    CA type: MT4123
    Number of ports: 2
    Firmware version: 20.31.1014
    Hardware version: 0
    Node GUID: 0x7cfe900300e1a820
    System image GUID: 0x7cfe900300e1a820
    Port 1:
        State: Active
        Physical state: LinkUp
        Rate: 200
        Base lid: 1
        LMC: 0
        SM lid: 1
        Link layer: InfiniBand
    Port 2:
        State: Down
        Physical state: Disabled
        Rate: 10
        Base lid: 0
        LMC: 0
        SM lid: 0
        Link layer: InfiniBand
"""
        device = self.scanner._parse_ibstat_device(output, "mlx5_0")
        self.assertIsNotNone(device)
        self.assertEqual(device.device_name, "mlx5_0")
        self.assertEqual(device.firmware_version, "20.31.1014")
        self.assertEqual(device.node_guid, "0x7cfe900300e1a820")
        self.assertEqual(device.num_ports, 2)
        self.assertEqual(len(device.ports), 2)

        # 验证端口1
        port1 = device.ports[0]
        self.assertEqual(port1.port_number, 1)
        self.assertEqual(port1.state, PortState.ACTIVE)
        self.assertEqual(port1.link_layer, LinkLayer.INFINIBAND)
        self.assertEqual(port1.effective_bandwidth_gbps, 200.0)
        self.assertEqual(port1.base_lid, 1)

        # 验证端口2
        port2 = device.ports[1]
        self.assertEqual(port2.port_number, 2)
        self.assertEqual(port2.state, PortState.DOWN)

    def test_parse_rdma_link(self):
        """测试解析rdma link show输出"""
        output = """link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev eth0
link mlx5_0/2 state DOWN physical_state DISABLED
link mlx5_1/1 state ACTIVE physical_state LINK_UP netdev eth1
"""
        links = self.scanner._parse_rdma_link(output)
        self.assertEqual(len(links), 3)
        self.assertIn("mlx5_0/1", links)
        self.assertEqual(links["mlx5_0/1"]["state"], "ACTIVE")
        self.assertEqual(links["mlx5_0/1"]["netdev"], "eth0")
        self.assertEqual(links["mlx5_0/2"]["state"], "DOWN")

    @patch.object(RDMADeviceScanner, '_run_remote_command')
    def test_scan_node_success(self, mock_run):
        """测试成功扫描节点"""
        # 模拟ibv_devices输出
        ibv_output = """    device                 node GUID
    ------              ----------------
    mlx5_0              7cfe900300e1a820
"""
        # 模拟ibstat输出
        ibstat_output = """CA 'mlx5_0'
    Number of ports: 1
    Firmware version: 20.31.1014
    Hardware version: 0
    Node GUID: 0x7cfe900300e1a820
    System image GUID: 0x7cfe900300e1a820
    Port 1:
        State: Active
        Physical state: LinkUp
        Rate: 200
        Base lid: 1
        SM lid: 1
        Link layer: InfiniBand
"""
        # 模拟rdma link输出
        rdma_output = "link mlx5_0/1 state ACTIVE physical_state LINK_UP\n"

        # 设置mock返回值序列
        mock_run.side_effect = [
            (ibv_output, "", 0),          # ibv_devices
            (ibstat_output, "", 0),       # ibstat mlx5_0
            ("0000:86:00.0", "", 0),      # PCI info
            ("5.8-1.0.4", "", 0),         # driver info
            (rdma_output, "", 0),          # rdma link show
        ]

        result = self.scanner.scan_node("node01", "192.168.1.1")

        self.assertTrue(result.scan_success)
        self.assertEqual(result.hostname, "node01")
        self.assertEqual(len(result.devices), 1)
        self.assertEqual(result.total_ports, 1)
        self.assertEqual(result.active_ports, 1)

    @patch.object(RDMADeviceScanner, '_run_remote_command')
    def test_scan_node_failure(self, mock_run):
        """测试节点扫描失败"""
        mock_run.return_value = ("", "Connection refused", 1)

        result = self.scanner.scan_node("node01", "192.168.1.1")

        self.assertFalse(result.scan_success)
        self.assertIn("ibv_devices执行失败", result.error_message)

    def test_get_cluster_summary(self):
        """测试集群摘要生成"""
        # 手动构建结果
        node_info = NodeRDMAInfo(
            hostname="node01",
            ip_address="192.168.1.1",
            scan_success=True,
            total_ports=2,
            active_ports=1,
        )
        node_info.devices = [
            RDMADevice(
                device_name="mlx5_0",
                node_guid="0x1234",
                system_image_guid="0x1234",
                firmware_version="20.31.1014",
                hardware_version="0",
                board_id="MT_123",
                num_ports=2,
                ports=[
                    RDMAPort(port_number=1, state=PortState.ACTIVE,
                             physical_state="LinkUp", link_layer=LinkLayer.INFINIBAND,
                             link_speed="200", link_width="4x",
                             effective_bandwidth_gbps=200.0),
                    RDMAPort(port_number=2, state=PortState.DOWN,
                             physical_state="Disabled", link_layer=LinkLayer.INFINIBAND,
                             link_speed="0", link_width="4x",
                             effective_bandwidth_gbps=0.0),
                ]
            )
        ]
        self.scanner.results = {"node01": node_info}

        summary = self.scanner.get_cluster_summary()
        self.assertEqual(summary["total_nodes"], 1)
        self.assertEqual(summary["successful_scans"], 1)
        self.assertEqual(summary["total_devices"], 1)
        self.assertEqual(summary["link_layers"]["InfiniBand"], 2)


class TestTopologyMapper(unittest.TestCase):
    """拓扑映射器测试"""

    def setUp(self):
        """测试初始化"""
        self.config = {
            "subnet_manager_node": "192.168.1.1",
            "ssh_user": "root",
            "ssh_key": "~/.ssh/id_rsa",
        }
        self.mapper = TopologyMapper(self.config)

    def test_parse_ibnetdiscover(self):
        """测试解析ibnetdiscover输出"""
        output = """Switch  36 "S-0002c90200400e30"  # "MF0;switch-01" enhanced port 0 lid 1
[1]  "H-7cfe900300e1a820"[1]    # "node01 HCA-1" lid 2
[2]  "H-7cfe900300e1a830"[1]    # "node02 HCA-1" lid 3

Ca  2 "H-7cfe900300e1a820"  # "node01 HCA-1"
[1]  "S-0002c90200400e30"[1]    # lid 1

Ca  2 "H-7cfe900300e1a830"  # "node02 HCA-1"
[1]  "S-0002c90200400e30"[2]    # lid 1
"""
        self.mapper._parse_ibnetdiscover(output)

        # 验证节点
        self.assertEqual(len(self.mapper.topology.nodes), 3)
        self.assertIn("S-0002c90200400e30", self.mapper.topology.nodes)
        self.assertIn("H-7cfe900300e1a820", self.mapper.topology.nodes)

        # 验证交换机
        switch = self.mapper.topology.nodes["S-0002c90200400e30"]
        self.assertEqual(switch.node_type, NodeType.SWITCH)
        self.assertEqual(switch.num_ports, 36)
        self.assertEqual(switch.lid, 1)

        # 验证主机
        host = self.mapper.topology.nodes["H-7cfe900300e1a820"]
        self.assertEqual(host.node_type, NodeType.HOST)

        # 验证链路
        self.assertGreater(len(self.mapper.topology.links), 0)

    def test_build_adjacency_graph(self):
        """测试构建邻接图"""
        # 添加测试数据
        self.mapper.topology.nodes = {
            "SW1": NetworkNode(guid="SW1", name="switch1", node_type=NodeType.SWITCH),
            "H1": NetworkNode(guid="H1", name="host1", node_type=NodeType.HOST),
            "H2": NetworkNode(guid="H2", name="host2", node_type=NodeType.HOST),
        }
        self.mapper.topology.links = [
            NetworkLink(source_guid="SW1", source_port=1, dest_guid="H1", dest_port=1),
            NetworkLink(source_guid="SW1", source_port=2, dest_guid="H2", dest_port=1),
        ]

        self.mapper._build_adjacency_graph()

        self.assertIn("H1", self.mapper.topology.adjacency["SW1"])
        self.assertIn("H2", self.mapper.topology.adjacency["SW1"])
        self.assertIn("SW1", self.mapper.topology.adjacency["H1"])

    def test_classify_switch_tiers(self):
        """测试交换机层级分类"""
        self.mapper.topology.nodes = {
            "SW_L1": NetworkNode(guid="SW_L1", name="leaf1", node_type=NodeType.SWITCH),
            "SW_L2": NetworkNode(guid="SW_L2", name="leaf2", node_type=NodeType.SWITCH),
            "SW_S1": NetworkNode(guid="SW_S1", name="spine1", node_type=NodeType.SWITCH),
            "H1": NetworkNode(guid="H1", name="host1", node_type=NodeType.HOST),
            "H2": NetworkNode(guid="H2", name="host2", node_type=NodeType.HOST),
        }
        # Leaf连接主机和Spine
        self.mapper.topology.adjacency = {
            "SW_L1": ["H1", "SW_S1"],
            "SW_L2": ["H2", "SW_S1"],
            "SW_S1": ["SW_L1", "SW_L2"],
            "H1": ["SW_L1"],
            "H2": ["SW_L2"],
        }

        self.mapper._classify_switch_tiers()

        self.assertEqual(self.mapper.topology.nodes["SW_L1"].tier, SwitchTier.LEAF)
        self.assertEqual(self.mapper.topology.nodes["SW_L2"].tier, SwitchTier.LEAF)
        self.assertEqual(self.mapper.topology.nodes["SW_S1"].tier, SwitchTier.SPINE)

    def test_get_shortest_path(self):
        """测试最短路径计算"""
        self.mapper.topology.adjacency = {
            "H1": ["SW1"],
            "SW1": ["H1", "SW2"],
            "SW2": ["SW1", "H2"],
            "H2": ["SW2"],
        }

        path = self.mapper.get_shortest_path("H1", "H2")
        self.assertEqual(path, ["H1", "SW1", "SW2", "H2"])
        self.assertEqual(self.mapper.get_hop_count("H1", "H2"), 3)

    def test_get_shortest_path_same_node(self):
        """测试同一节点路径"""
        path = self.mapper.get_shortest_path("H1", "H1")
        self.assertEqual(path, ["H1"])

    def test_identify_spine_leaf_topology(self):
        """测试Spine-Leaf拓扑识别"""
        self.mapper.topology.nodes = {
            "L1": NetworkNode(guid="L1", name="leaf1", node_type=NodeType.SWITCH, tier=SwitchTier.LEAF),
            "L2": NetworkNode(guid="L2", name="leaf2", node_type=NodeType.SWITCH, tier=SwitchTier.LEAF),
            "S1": NetworkNode(guid="S1", name="spine1", node_type=NodeType.SWITCH, tier=SwitchTier.SPINE),
            "S2": NetworkNode(guid="S2", name="spine2", node_type=NodeType.SWITCH, tier=SwitchTier.SPINE),
            "H1": NetworkNode(guid="H1", name="host1", node_type=NodeType.HOST),
            "H2": NetworkNode(guid="H2", name="host2", node_type=NodeType.HOST),
        }
        self.mapper.topology.switch_tiers = {
            "leaf": ["L1", "L2"],
            "spine": ["S1", "S2"],
            "core": [],
        }
        # 每个Leaf连接所有Spine
        self.mapper.topology.adjacency = {
            "L1": ["H1", "S1", "S2"],
            "L2": ["H2", "S1", "S2"],
            "S1": ["L1", "L2"],
            "S2": ["L1", "L2"],
            "H1": ["L1"],
            "H2": ["L2"],
        }

        self.mapper._identify_topology_type()
        self.assertEqual(self.mapper.topology.topology_type, TopologyType.SPINE_LEAF)

    def test_export_dot(self):
        """测试DOT格式导出"""
        self.mapper.topology.nodes = {
            "SW1": NetworkNode(guid="SW1", name="switch1", node_type=NodeType.SWITCH,
                               tier=SwitchTier.LEAF),
            "H1": NetworkNode(guid="H1", name="host1", node_type=NodeType.HOST),
        }
        self.mapper.topology.links = [
            NetworkLink(source_guid="SW1", source_port=1, dest_guid="H1", dest_port=1),
        ]

        dot_output = self.mapper.export_dot()
        self.assertIn("graph cluster_topology", dot_output)
        self.assertIn("SW1", dot_output)
        self.assertIn("H1", dot_output)

    def test_export_json(self):
        """测试JSON格式导出"""
        self.mapper.topology.nodes = {
            "SW1": NetworkNode(guid="SW1", name="switch1", node_type=NodeType.SWITCH),
        }
        self.mapper.topology.links = []
        self.mapper.topology.topology_type = TopologyType.SPINE_LEAF
        self.mapper.topology.num_hosts = 0
        self.mapper.topology.num_switches = 1
        self.mapper.topology.num_links = 0

        json_data = self.mapper.export_json()
        self.assertEqual(json_data["topology_type"], "spine_leaf")
        self.assertEqual(json_data["num_switches"], 1)
        self.assertEqual(len(json_data["nodes"]), 1)


if __name__ == "__main__":
    unittest.main()
