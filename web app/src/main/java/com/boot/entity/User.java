package com.boot.entity;

import java.io.Serializable;
import java.util.Date;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Table;

import com.fasterxml.jackson.annotation.JsonFormat;

import lombok.Data;

@Entity
@Table(name = "t_user")
@Data
public class User implements Serializable{
	private static final long serialVersionUID = -4298309205022284451L;
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Integer id;
	private String name;
	private String username;//登录账号
	private String password;//密码
	
	@Column(name="create_time")
	private Date createTime = new Date();//创建时间
	
	@Column(name="update_time")
	@JsonFormat(pattern="yyyy-MM-dd HH:mm")
	private Date updateTime = new Date();
	
	private Integer sex;//性别 0男1女
	private Integer age;//年龄
	private String phone;//电话
	private String bio;
	
	@Column(name="last_time")
	@JsonFormat(pattern="yyyy-MM-dd HH:mm")
	private Date lastTime;
	
}
